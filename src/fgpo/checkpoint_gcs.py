"""Cell-scoped immutable GCS snapshots with bounded streaming I/O.

CURRENT is the only training authority. Actor files alone are materialized; the
unchanged PyTorch objects use a non-seekable writer and generation-pinned reader.
The operator refreshes FGPO_GCS_ACCESS_TOKEN_FILE outside this process.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import http.client
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
from urllib.parse import quote, urlencode, urlsplit
import uuid

from .records import (_cost_tail_id, _merge_tail_observations,
                      _read_cost_tail_state, _write_cost_tail_state)

MANIFEST_FORMAT = "fgpo-gcs-manifest-v2"
SUPPORTED_MANIFEST_FORMATS = frozenset({"fgpo-gcs-manifest-v1", MANIFEST_FORMAT})
JOURNAL_FORMAT = "fgpo-gcs-journal-v2"
LOGS_METADATA_FORMAT = "fgpo-gcs-logs-v2"
SUPPORTED_LOGS_METADATA_FORMATS = frozenset({"fgpo-gcs-logs-v1", LOGS_METADATA_FORMAT})
POINTER_FORMAT = "fgpo-gcs-pointer-v1"
READER_PIN_FORMAT = "fgpo-gcs-reader-pin-v1"
CHUNK_BYTES = 8 * 1024 * 1024
JSON_LIMIT = 16 * 1024 * 1024
_JOURNAL_COALESCE_FLOOR = 64 * 1024
_TRANSFER_CATEGORIES = ("actor", "optimizer", "gradients", "reference", "teacher", "state", "artifacts", "journals")


class _PointerSuperseded(Exception):
    """Mutable GCS pointer moved between metadata stat and pointer read."""


class StorageError(RuntimeError):
    """Only operation/status escape this module; HTTP bodies/URLs never do."""
    def __init__(self, operation, status):
        self.operation, self.status = operation, status
        super().__init__(f"GCS {operation} failed (status={status})")


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def _relative(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("GCS manifest member must be a canonical relative path")
    return value


class GCSClient:
    """Standard-library JSON API client, with no retry or credential discovery."""
    def __init__(self, bucket, *, chunk_bytes=CHUNK_BYTES, upload_workers=8):
        self.bucket = bucket
        self.chunk_bytes = int(chunk_bytes)
        self.upload_workers = max(1, int(upload_workers))
        self._connection = None
        self._operation_depth = 0
        if not bucket or "/" in bucket or self.chunk_bytes < 256 * 1024 or self.chunk_bytes % (256 * 1024):
            raise ValueError("GCS chunks must be a positive multiple of 256 KiB")

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    @contextmanager
    def operation(self):
        """Reuse TLS within one operation; never retain it into training work."""
        outermost = self._operation_depth == 0
        if outermost:
            self.close()
        self._operation_depth += 1
        try:
            yield
        finally:
            self._operation_depth -= 1
            if outermost:
                self.close()

    def _token(self):
        try:
            path = Path(os.environ["FGPO_GCS_ACCESS_TOKEN_FILE"])
            with path.open("r", encoding="utf-8") as handle:
                mode = os.fstat(handle.fileno())
                if (not stat.S_ISREG(mode.st_mode) or mode.st_uid != os.getuid()
                        or mode.st_mode & 0o077 or mode.st_size > 16384):
                    raise ValueError()
                token = handle.read().strip()
            if not token or any(character.isspace() for character in token):
                raise ValueError()
            return token
        except (KeyError, OSError, ValueError):
            raise StorageError("read_access_token", "restricted_token_file_required") from None

    @contextmanager
    def response(self, method, url, *, operation, body=None, headers=None, statuses=(200,)):
        with self.operation():
            with self._response(method, url, operation=operation, body=body, headers=headers, statuses=statuses) as response:
                yield response

    @contextmanager
    def _response(self, method, url, *, operation, body=None, headers=None, statuses=(200,)):
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.hostname != "storage.googleapis.com" or parts.port not in (None, 443):
            raise StorageError(operation, "invalid_endpoint")
        if self._connection is None:
            self._connection = http.client.HTTPSConnection(parts.hostname, timeout=600)
        connection = self._connection
        try:
            values = {"Authorization": "Bearer " + self._token(), **(headers or {})}
            connection.request(method, parts.path + ("?" + parts.query if parts.query else ""), body=body, headers=values)
            response = connection.getresponse()
            if response.status not in statuses:
                raise StorageError(operation, response.status)
            yield response
            # Drain small upload acknowledgements; media readers consume their
            # own bodies. Reuse TLS for the thousands of bounded Adam chunks.
            if len(response.read(JSON_LIMIT + 1)) > JSON_LIMIT:
                raise StorageError(operation, "response_tail_too_large")
        except StorageError:
            connection.close()
            self._connection = None
            raise
        except Exception:
            connection.close()
            self._connection = None
            raise StorageError(operation, "transport_or_response_error") from None

    def _url(self, name=None, **params):
        base = "https://storage.googleapis.com/storage/v1/b/" + quote(self.bucket, safe="") + "/o"
        if name is not None:
            base += "/" + quote(name, safe="")
        return base + ("?" + urlencode(params) if params else "")

    def _json(self, response):
        raw = response.read(JSON_LIMIT + 1)
        if len(raw) > JSON_LIMIT:
            raise StorageError("read_json", "response_too_large")
        try:
            return json.loads(raw)
        except (ValueError, UnicodeError):
            raise StorageError("read_json", "invalid_json") from None

    def metadata(self, name, *, optional=False):
        try:
            with self.response("GET", self._url(name), operation="object_metadata") as response:
                return self._json(response)
        except StorageError as error:
            if optional and error.status == 404:
                return None
            raise

    def descriptor(self, metadata):
        return {"name": metadata["name"], "generation": str(metadata["generation"]), "size": int(metadata["size"])}

    def read_json(self, descriptor):
        if int(descriptor["size"]) > JSON_LIMIT:
            raise ValueError("GCS JSON member exceeds the metadata bound")
        with self.response("GET", self._url(descriptor["name"], alt="media", generation=descriptor["generation"]),
                           operation="read_json_object") as response:
            return self._json(response)

    def write_json(self, name, value, *, generation=0):
        url = "https://storage.googleapis.com/upload/storage/v1/b/" + quote(self.bucket, safe="") + "/o?"
        url += urlencode({"uploadType": "media", "name": name, "ifGenerationMatch": generation})
        with self.response("POST", url, operation="publish_json", body=_json_bytes(value),
                           headers={"Content-Type": "application/json"}, statuses=(200, 201)) as response:
            return self.descriptor(self._json(response))

    def writer(self, name):
        return GCSWriter(self, name)

    def compose(self, name, sources):
        if not sources or len(sources) > 32:
            raise ValueError("GCS compose requires 1 to 32 source objects")
        url = (
            "https://storage.googleapis.com/storage/v1/b/"
            + quote(self.bucket, safe="")
            + "/o/"
            + quote(name, safe="")
            + "/compose?"
            + urlencode({"ifGenerationMatch": 0})
        )
        body = {
            "sourceObjects": [
                {"name": item["name"], "generation": str(item["generation"])}
                for item in sources
            ],
            "destination": {"contentType": "application/octet-stream"},
        }
        with self.response(
            "POST",
            url,
            operation="compose_object",
            body=_json_bytes(body),
            headers={"Content-Type": "application/json"},
            statuses=(200,),
        ) as response:
            return self.descriptor(self._json(response))

    def upload_file(self, name, path):
        path = Path(path)
        size = path.stat().st_size
        parallel_floor = max(256 * 1024 * 1024, 2 * self.chunk_bytes)
        if self.upload_workers <= 1 or size < parallel_floor:
            return self.upload_range(name, path, offset=0, size=size)
        part_count = min(24, max(self.upload_workers, math.ceil(size / self.chunk_bytes)))
        base_part, rem = divmod(size, part_count)
        ranges = []
        offset = 0
        for index in range(part_count):
            part_size = base_part + (1 if index < rem else 0)
            ranges.append((index, offset, part_size))
            offset += part_size
        temp_prefix = f"{name}.fgpo-parts/{uuid.uuid4().hex}"
        uploaded = [None] * part_count

        def _upload_part(item):
            index, part_offset, part_size = item
            worker = GCSClient(self.bucket, chunk_bytes=self.chunk_bytes, upload_workers=1)
            try:
                desc = worker.upload_range(f"{temp_prefix}/{index:02d}", path, offset=part_offset, size=part_size)
                uploaded[index] = desc
                return desc
            finally:
                worker.close()

        def _delete_part(desc):
            if desc is None:
                return
            worker = GCSClient(self.bucket, chunk_bytes=self.chunk_bytes, upload_workers=1)
            try:
                worker.delete(desc)
            except StorageError:
                pass
            finally:
                worker.close()

        try:
            with ThreadPoolExecutor(max_workers=min(self.upload_workers, part_count)) as pool:
                descriptors = list(pool.map(_upload_part, ranges))
            return self.compose(name, descriptors)
        finally:
            cleanup = [desc for desc in uploaded if desc is not None]
            if cleanup:
                with ThreadPoolExecutor(max_workers=min(self.upload_workers, len(cleanup))) as pool:
                    list(pool.map(_delete_part, cleanup))

    def upload_range(self, name, path, *, offset=0, size):
        remaining = int(size)
        if offset < 0 or remaining < 0:
            raise ValueError("GCS upload offset and size must be nonnegative")
        with self.operation():
            writer = self.writer(name)
            with Path(path).open("rb") as source:
                source.seek(int(offset))
                while remaining:
                    block = source.read(min(self.chunk_bytes, remaining))
                    if not block:
                        raise StorageError("upload_range", "source_short_read")
                    writer.write(block)
                    remaining -= len(block)
            return writer.finish()

    def reader(self, descriptor):
        return GCSRangeReader(self, descriptor)

    def download_file(self, descriptor, path):
        self.download_segments((descriptor,), path, expected_size=int(descriptor["size"]))

    def download_segments(self, segments, path, *, expected_size):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(path.name + ".incoming")
        try:
            with pending.open("wb") as target:
                total = 0
                for descriptor in segments:
                    with self.response("GET", self._url(descriptor["name"], alt="media", generation=descriptor["generation"]),
                                       operation="download_actor_or_journal") as response:
                        size = 0
                        while True:
                            block = response.read(self.chunk_bytes)
                            if not block:
                                break
                            target.write(block)
                            size += len(block)
                        if size != int(descriptor["size"]):
                            raise StorageError("download_actor_or_journal", "short_read")
                        total += size
                if total != int(expected_size):
                    raise StorageError("download_actor_or_journal", "journal_length_mismatch")
                target.flush()
                os.fsync(target.fileno())
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)

    def list_objects(self, prefix):
        page = None
        while True:
            params = {"prefix": prefix, "fields": "items(name,generation,size),nextPageToken", "maxResults": 1000}
            if page:
                params["pageToken"] = page
            with self.response("GET", self._url(**params), operation="list_objects") as response:
                result = self._json(response)
            yield from (self.descriptor(item) for item in result.get("items", ()))
            page = result.get("nextPageToken")
            if not page:
                break

    def delete(self, descriptor):
        with self.response("DELETE", self._url(descriptor["name"], generation=descriptor["generation"],
                           ifGenerationMatch=descriptor["generation"]), operation="delete_unreferenced_object", statuses=(204,)):
            pass


class GCSWriter:
    """Keep one final chunk pending so unknown-length torch.save can finalize."""
    def __init__(self, client, name):
        self.client, self.name = client, name
        self.buffer = bytearray()
        self.position = self.sent = 0
        self.result = None
        self.failed = False
        url = "https://storage.googleapis.com/upload/storage/v1/b/" + quote(client.bucket, safe="") + "/o?"
        url += urlencode({"uploadType": "resumable", "name": name, "ifGenerationMatch": 0})
        with client.response("POST", url, operation="initiate_upload", body=b"",
                             headers={"X-Upload-Content-Type": "application/octet-stream"}) as response:
            self._session_url = response.getheader("Location")
        if not self._session_url:
            raise StorageError("initiate_upload", "missing_location")

    def __repr__(self):
        return "<GCSWriter bounded streaming upload>"

    def writable(self):
        return True

    def seekable(self):
        return False

    def tell(self):
        return self.position

    def write(self, data):
        if self.failed:
            raise StorageError("upload_aborted", "previous_failure")
        if self.result is not None:
            raise ValueError("Upload is already finalized")
        view = memoryview(data).cast("B")
        length = len(view)
        while view:
            if len(self.buffer) == self.client.chunk_bytes:
                self._send(final=False)
            count = min(len(view), self.client.chunk_bytes - len(self.buffer))
            self.buffer.extend(view[:count])
            view = view[count:]
        self.position += length
        return length

    def flush(self):
        # torch.save flush is not EOF; retain the last bounded chunk.
        pass

    def _send(self, *, final):
        if self.failed:
            raise StorageError("upload_aborted", "previous_failure")
        size = len(self.buffer)
        total = str(self.sent + size) if final else "*"
        content_range = f"bytes {self.sent}-{self.sent + size - 1}/{total}" if size else f"bytes */{total}"
        try:
            with self.client.response("PUT", self._session_url, operation="upload_final_chunk" if final else "upload_chunk",
                                      body=self.buffer, headers={"Content-Type": "application/octet-stream",
                                      "Content-Range": content_range, "Content-Length": str(size)},
                                      statuses=(200, 201) if final else (308,)) as response:
                if final:
                    self.result = self.client.descriptor(self.client._json(response))
                    if self.result["name"] != self.name or self.result["size"] != self.sent + size:
                        raise StorageError("upload_final_chunk", "object_identity_or_length")
                elif response.getheader("Range") != f"bytes=0-{self.sent + size - 1}":
                    raise StorageError("upload_chunk", "unexpected_committed_range")
        except BaseException:
            # torch.save may write ZIP cleanup records after an exception.
            # Latch the failure so cleanup cannot resend a failed chunk.
            self.failed = True
            self._session_url = None
            raise
        self.sent += size
        self.buffer.clear()

    def finish(self):
        if self.failed:
            raise StorageError("upload_aborted", "previous_failure")
        if self.result is None:
            self._send(final=True)
            self._session_url = None
        return self.result


class GCSRangeReader(io.RawIOBase):
    """No whole-object cache; readinto fills caller storage in bounded chunks."""
    def __init__(self, client, descriptor):
        super().__init__()
        self.client, self.descriptor, self.position = client, descriptor, 0
        self.size = int(descriptor["size"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = offset if whence == io.SEEK_SET else self.position + offset if whence == io.SEEK_CUR else self.size + offset if whence == io.SEEK_END else -1
        if position < 0:
            raise ValueError("Invalid remote object seek")
        self.position = position
        return position

    def readinto(self, destination):
        view = memoryview(destination).cast("B")
        count = min(len(view), max(0, self.size - self.position))
        if not count:
            return 0
        start, end = self.position, self.position + count - 1
        url = self.client._url(self.descriptor["name"], alt="media", generation=self.descriptor["generation"])
        with self.client.response("GET", url, operation="range_read", headers={"Range": f"bytes={start}-{end}"},
                                  statuses=(206,)) as response:
            if response.getheader("Content-Range") != f"bytes {start}-{end}/{self.size}":
                raise StorageError("range_read", "unexpected_content_range")
            consumed = 0
            while consumed < count:
                amount = response.readinto(view[consumed:min(count, consumed + self.client.chunk_bytes)])
                if not amount:
                    raise StorageError("range_read", "short_read")
                consumed += amount
        self.position += count
        return count

    def read(self, size=-1):
        # Archive headers use read(); tensor payloads use PyTorch's readinto
        # adapter. Reject any caller attempting a whole-object allocation.
        if size is None or size < 0 or size > self.client.chunk_bytes:
            raise io.UnsupportedOperation("Large or unbounded GCS reads require readinto")
        size = min(size, max(0, self.size - self.position))
        result = bytearray(size)
        count = self.readinto(result)
        return bytes(result) if count == size else bytes(memoryview(result)[:count])


def configured(config):
    backend = config.get("checkpoint_backend", "local")
    if backend not in {"local", "gcs"}:
        raise ValueError("checkpoint_backend must be local or gcs")
    return backend == "gcs"


def _committed_jsonl_size(path):
    """Return byte offset through the final complete newline-terminated line."""
    with Path(path).open("rb") as handle:
        handle.seek(0, 2)
        end = handle.tell()
        if not end:
            return 0
        handle.seek(end - 1)
        if handle.read(1) == b"\n":
            return end
        while end:
            start = max(0, end - 8192)
            handle.seek(start)
            boundary = handle.read(end - start).rfind(b"\n")
            if boundary >= 0:
                return start + boundary + 1
            end = start
        return 0


def _manifest_journal(manifest, name):
    """Read a v2 incremental journal or lift an existing v1 single-object entry."""
    if manifest is None:
        return None
    journal = manifest.get("journals", {}).get(name)
    if journal is not None:
        return journal
    descriptor = manifest.get("artifacts", {}).get(name)
    if descriptor is None:
        return None
    size = int(descriptor["size"])
    segments = [] if size == 0 else [{"offset": 0, **descriptor}]
    return {
        "format": JOURNAL_FORMAT,
        "committed_bytes": size,
        "lineage_id": f"v1-{manifest['snapshot']}",
        "segments": segments,
    }


def _pending_cost_tail(journal):
    size = int(journal.get("uncommitted_tail_bytes_at_publish", 0))
    if not size:
        return None
    observation = journal.get("pending_tail_observation")
    return {"id": (observation["id"] if observation is not None
                   else _cost_tail_id(None, journal["committed_bytes"])),
            "bytes": size}


def _select_costs_journal(*manifests):
    """Select the longest physical-cost journal while merging equal-prefix tail evidence."""
    candidates = []
    for manifest in manifests:
        cand = _manifest_journal(manifest, "costs.jsonl")
        if cand is not None:
            candidates.append(cand)
    if not candidates:
        return None
    consumed = _merge_tail_observations(
        *(item.get("consumed_tail_observations", {}) for item in candidates))

    def selection_key(item):
        observation = _pending_cost_tail(item)
        outstanding = (0 if observation is None else
                       max(0, observation["bytes"] - consumed.get(observation["id"], 0)))
        return int(item["committed_bytes"]), outstanding

    best = max(candidates, key=selection_key)
    max_discarded = max(int(item.get("discarded_incomplete_tail_bytes", 0)) for item in candidates)
    merged = dict(best)
    merged["discarded_incomplete_tail_bytes"] = max_discarded
    observation = _pending_cost_tail(best)
    if observation is not None and consumed.get(observation["id"], 0) >= observation["bytes"]:
        observation = None
    merged["pending_tail_observation"] = observation
    merged["uncommitted_tail_bytes_at_publish"] = 0 if observation is None else observation["bytes"]
    merged["consumed_tail_observations"] = consumed
    return merged


def _restore_cost_tail_provenance(path, journal, *, abandon_pending, clear_writer=False):
    state = _read_cost_tail_state(path)
    consumed = _merge_tail_observations(
        state["consumed_tail_observations"], journal.get("consumed_tail_observations", {}))
    count = max(state["abandoned_costs_tail_bytes"],
                int(journal.get("discarded_incomplete_tail_bytes", 0)))
    observation = _pending_cost_tail(journal)
    if abandon_pending and observation is not None:
        identity, size = observation["id"], observation["bytes"]
        count += max(0, size - consumed.get(identity, 0))
        consumed[identity] = max(size, consumed.get(identity, 0))
    if clear_writer:
        _write_cost_tail_state(path, count=count, consumed=consumed, writer_session_id=None)
    else:
        _write_cost_tail_state(path, count=count, consumed=consumed)


def _empty_transfer_stats():
    return {
        "uploaded_bytes": 0,
        "reused_bytes": 0,
        "uploaded_objects": 0,
        "reused_objects": 0,
        "metadata_uploaded_bytes": 0,
        "metadata_uploaded_objects": 0,
        "by_category": {
            cat: {"uploaded_bytes": 0, "reused_bytes": 0, "uploaded_objects": 0, "reused_objects": 0}
            for cat in _TRANSFER_CATEGORIES
        },
    }


def _add_transfer(stats, category, descriptors, *, uploaded):
    if isinstance(descriptors, dict):
        descriptors = (descriptors,)
    total_bytes = sum(int(d["size"]) for d in descriptors)
    count = len(descriptors)
    bucket = stats["by_category"][category]
    if uploaded:
        stats["uploaded_bytes"] += total_bytes
        stats["uploaded_objects"] += count
        bucket["uploaded_bytes"] += total_bytes
        bucket["uploaded_objects"] += count
    else:
        stats["reused_bytes"] += total_bytes
        stats["reused_objects"] += count
        bucket["reused_bytes"] += total_bytes
        bucket["reused_objects"] += count


def _merge_transfer_stats(target, extra):
    for key in ("uploaded_bytes", "reused_bytes", "uploaded_objects", "reused_objects",
                "metadata_uploaded_bytes", "metadata_uploaded_objects"):
        target[key] += int(extra.get(key, 0))
    for cat in _TRANSFER_CATEGORIES:
        src = extra.get("by_category", {}).get(cat)
        if src:
            dst = target["by_category"][cat]
            for key in ("uploaded_bytes", "reused_bytes", "uploaded_objects", "reused_objects"):
                dst[key] += int(src.get(key, 0))


def _manifest_descriptors(manifest, manifest_descriptor=None):
    refs = set()
    if manifest_descriptor is not None:
        refs.add((manifest_descriptor["name"], manifest_descriptor["generation"]))
    if manifest is None:
        return refs
    for category in ("actor", "tensors", "artifacts"):
        refs.update((item["name"], item["generation"]) for item in manifest.get(category, {}).values())
    for journal in manifest.get("journals", {}).values():
        refs.update((seg["name"], seg["generation"]) for seg in journal.get("segments", ()))
    return refs


class GCSCheckpoint:
    def __init__(self, config, root):
        uri = urlsplit(config["checkpoint_gcs_uri"])
        if uri.scheme != "gs" or not uri.netloc or uri.query or uri.fragment:
            raise ValueError("checkpoint_gcs_uri must name a gs://bucket/study prefix")
        self.root = Path(root)
        self.cell = self.root.parent
        self.client = GCSClient(
            uri.netloc,
            chunk_bytes=int(config.get("checkpoint_gcs_chunk_mib", 8)) * 1024 * 1024,
            upload_workers=int(config.get("checkpoint_gcs_upload_workers", 8)),
        )
        suffix = f"{config['suite']}/{config['benchmark']}/{config['method']}/seed-{int(config['seed'])}"
        self.prefix = uri.path.strip("/") + "/" + suffix
        self.identity = {key: config[key] for key in ("protocol", "suite", "benchmark", "method", "seed", "model_path")}
        self.config = config
        self.last_transfer_stats = None
        self._transaction_depth = 0

    @contextmanager
    def transaction(self):
        if self._transaction_depth > 0:
            self._transaction_depth += 1
            try:
                with self.client.operation():
                    yield
            finally:
                self._transaction_depth -= 1
            return
        self.cell.mkdir(parents=True, exist_ok=True)
        with (self.cell / ".gcs-transaction.lock").open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            self._transaction_depth += 1
            try:
                with self.client.operation():
                    yield
            finally:
                self._transaction_depth -= 1
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _cell_lock_if_present(self):
        """Share the publishing runtime's cell transaction lock when running on-host."""
        if self._transaction_depth == 0 and self.cell.is_dir() and (self.cell / ".gcs-transaction.lock").is_file():
            with self.transaction():
                yield
        else:
            with self.client.operation():
                yield

    def _read_pointer_ref(self, pointer, *, optional=True):
        metadata = self.client.metadata(self.prefix + "/" + pointer, optional=optional)
        if metadata is None:
            return None, None
        descriptor = self.client.descriptor(metadata)
        try:
            value = self.client.read_json(descriptor)
        except StorageError as error:
            if error.status == 404:
                raise _PointerSuperseded() from None
            raise
        if value.get("format") != POINTER_FORMAT:
            raise ValueError("GCS pointer format differs from this exact study cell")
        self._check_descriptor(value["manifest"])
        return descriptor, value["manifest"]

    def _release_pin(self, pin_desc):
        try:
            self.client.delete(pin_desc)
        except StorageError as error:
            if error.status != 404:
                raise

    @contextmanager
    def _reader_pins(self, pointers=("CURRENT", "FINAL", "LOGS"), *, required=(), verify_cell_root=False):
        """Acquire linearizable GCS reader pins before fetching manifests or members.

        Ordering proof against concurrent publishers:
        1) Reader reads current pointer generations and manifest descriptors
           (restarting cleanly on _PointerSuperseded if a mutable pointer moves).
        2) Reader writes a reader pin under readers/<lease>.json (ifGenerationMatch=0).
        3) Every subsequent step—pointer confirmation, manifest validation, and
           caller member reads—is wrapped in cleanup so any failure deletes the pin.
        4) Reader re-reads pointer metadata generations; if any pointer moved, the
           pin is deleted and acquisition repeats without consuming stale members.
        """
        required = set(required)
        pin_desc = None
        resolved = None
        try:
            for _ in range(16):
                try:
                    refs = {ptr: self._read_pointer_ref(ptr, optional=(ptr not in required)) for ptr in pointers}
                except _PointerSuperseded:
                    continue
                manifest_refs = [m_desc for _, m_desc in refs.values() if m_desc is not None]
                if not manifest_refs:
                    resolved = {ptr: (None, None, None) for ptr in pointers}
                    break
                lease_id = uuid.uuid4().hex
                pin_desc = self.client.write_json(
                    self.prefix + f"/readers/{lease_id}.json",
                    {"format": READER_PIN_FORMAT, "lease_id": lease_id, "manifests": manifest_refs},
                    generation=0,
                )
                stable = True
                for ptr, (ptr_desc, _) in refs.items():
                    meta = self.client.metadata(self.prefix + "/" + ptr, optional=True)
                    if ptr_desc is None:
                        if meta is not None:
                            stable = False
                            break
                    elif meta is None or str(meta["generation"]) != ptr_desc["generation"]:
                        stable = False
                        break
                if not stable:
                    self._release_pin(pin_desc)
                    pin_desc = None
                    continue
                resolved = {}
                for ptr, (ptr_desc, m_desc) in refs.items():
                    if m_desc is None:
                        resolved[ptr] = (None, None, None)
                    else:
                        manifest = self._validate_manifest(
                            self.client.read_json(m_desc), verify_cell_root=verify_cell_root
                        )
                        resolved[ptr] = (manifest, ptr_desc, m_desc)
                break
            else:
                raise StorageError("acquire_reader_pin", "concurrent_pointer_churn")
            yield resolved
        finally:
            if pin_desc is not None:
                self._release_pin(pin_desc)

    @contextmanager
    def open_snapshot(self, pointers=("CURRENT", "FINAL", "LOGS"), *, required=(), verify_cell_root=True):
        """Hold cell lock (when on-host) and GCS reader pins across the entire read."""
        with self._cell_lock_if_present(), self._reader_pins(
                pointers, required=required, verify_cell_root=verify_cell_root) as pinned:
            yield pinned

    def _active_reader_pins(self):
        pins = []
        for pin_desc in self.client.list_objects(self.prefix + "/readers/"):
            try:
                pin_data = self.client.read_json(pin_desc)
            except StorageError as error:
                if error.status == 404:
                    continue
                raise
            if pin_data.get("format") != READER_PIN_FORMAT:
                raise ValueError("Invalid GCS reader pin format")
            pins.append((pin_desc, pin_data))
        return pins

    def current(self, pointer="CURRENT", *, optional=False):
        if self._transaction_depth > 0:
            return self._current(pointer, optional=optional, verify_cell_root=True)
        required = () if optional else (pointer,)
        with self.open_snapshot((pointer,), required=required, verify_cell_root=True) as pinned:
            manifest, pointer_desc, _ = pinned[pointer]
            return manifest, pointer_desc

    def inspect(self, pointer="CURRENT", *, optional=False):
        """Read and validate a cell manifest without requiring a matching local cell_root."""
        if self._transaction_depth > 0:
            return self._current_entry(pointer, optional=optional, verify_cell_root=False)
        required = () if optional else (pointer,)
        with self.open_snapshot((pointer,), required=required, verify_cell_root=False) as pinned:
            return pinned[pointer]

    def _validate_manifest(self, manifest, *, verify_cell_root=True):
        if manifest.get("format") not in SUPPORTED_MANIFEST_FORMATS or manifest.get("identity") != self.identity:
            raise ValueError("GCS snapshot identity differs from this exact study cell")
        if verify_cell_root and manifest.get("cell_root") != str(self.cell.resolve()):
            raise ValueError("GCS snapshot cell_root differs from this exact study cell")
        self._check_members(manifest)
        return manifest

    def _current(self, pointer, *, optional, verify_cell_root=True):
        manifest, pointer_desc, _ = self._current_entry(pointer, optional=optional, verify_cell_root=verify_cell_root)
        return manifest, pointer_desc

    def _current_entry(self, pointer, *, optional, verify_cell_root=True):
        descriptor, manifest_desc = self._read_pointer_ref(pointer, optional=optional)
        if descriptor is None:
            return None, None, None
        manifest = self._validate_manifest(self.client.read_json(manifest_desc), verify_cell_root=verify_cell_root)
        return manifest, descriptor, manifest_desc

    def _check_descriptor(self, value):
        if (not isinstance(value, dict) or not str(value.get("name", "")).startswith(self.prefix + "/")
                or not str(value.get("generation", "")).isdigit()
                or type(value.get("size")) is not int or value["size"] < 0):
            raise ValueError("GCS object descriptor escapes this cell or lacks a fixed generation/length")

    def _check_journal(self, name, journal):
        if (not isinstance(journal, dict) or journal.get("format") != JOURNAL_FORMAT
                or type(journal.get("committed_bytes")) is not int or journal["committed_bytes"] < 0
                or not isinstance(journal.get("lineage_id"), str) or not journal["lineage_id"]
                or not isinstance(journal.get("segments"), list)):
            raise ValueError(f"GCS incremental journal {name!r} has an invalid header")
        for opt_key in ("uncommitted_tail_bytes_at_publish", "discarded_incomplete_tail_bytes"):
            if opt_key in journal and (type(journal[opt_key]) is not int or journal[opt_key] < 0):
                raise ValueError(f"GCS incremental journal {name!r} has invalid {opt_key}")
        committed = journal["committed_bytes"]
        segments = journal["segments"]
        if committed == 0:
            if segments:
                raise ValueError(f"Empty GCS journal {name!r} cannot contain segments")
            return
        expected_offset = 0
        for seg in segments:
            self._check_descriptor(seg)
            if type(seg.get("offset")) is not int or seg["offset"] != expected_offset or seg["size"] <= 0:
                raise ValueError(f"GCS journal {name!r} segments are non-contiguous or empty")
            expected_offset += seg["size"]
        if expected_offset != committed:
            raise ValueError(f"GCS journal {name!r} committed_bytes disagrees with segment total")

    def _check_members(self, manifest):
        for category in ("actor", "tensors", "artifacts"):
            for name, value in manifest.get(category, {}).items():
                _relative(name)
                self._check_descriptor(value)
        for name, journal in manifest.get("journals", {}).items():
            _relative(name)
            self._check_journal(name, journal)

    def _new_manifest(self, snapshot, metadata, actor, tensors, artifacts, inventory, journals=None):
        return {"format": MANIFEST_FORMAT, "snapshot": snapshot, "identity": self.identity,
                "cell_root": str(self.cell.resolve()), "metadata": metadata, "actor": actor,
                "tensors": tensors, "artifacts": artifacts,
                "journals": {} if journals is None else journals, "inventory": inventory}

    def _journal_state_path(self):
        return self.cell / ".gcs-journal-state.json"

    def _read_local_journal_state(self):
        path = self._journal_state_path()
        if not path.is_file():
            return {"artifacts": {}, "journals": {}, "abandoned_costs_tail_bytes": 0}
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "artifacts": dict(data.get("artifacts", {})),
            "journals": dict(data.get("journals", {})),
            "abandoned_costs_tail_bytes": int(data.get("abandoned_costs_tail_bytes", 0)),
        }

    def _write_local_journal_state(self, state):
        path = self._journal_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(path.name + ".incoming")
        with pending.open("wb") as handle:
            handle.write(_json_bytes(state))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)

    def _reset_training_lineage(self, manifest):
        state = self._read_local_journal_state()
        entry = _manifest_journal(manifest, "training.jsonl")
        state["journals"]["training.jsonl"] = {
            "lineage_id": uuid.uuid4().hex,
            "committed_bytes": 0 if entry is None else int(entry["committed_bytes"]),
        }
        self._write_local_journal_state(state)

    def _local_pointer(self, manifest, descriptor, pointer="CURRENT"):
        from .checkpoint import _sync_directory
        directory = self.cell / ("final_checkpoint" if pointer == "FINAL" else "checkpoint")
        directory.mkdir(parents=True, exist_ok=True)
        value = {"format": POINTER_FORMAT, "backend": "gcs", "snapshot": manifest["snapshot"],
                 "bucket": self.client.bucket, "prefix": self.prefix, "remote_pointer": pointer,
                 "manifest": descriptor}
        pending = directory / "CURRENT.incoming"
        with pending.open("wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, directory / "CURRENT")
        _sync_directory(directory)

    def _publish(self, manifest, previous, *, pointer="CURRENT", local_pointer=True, known_manifests=None,
                 stats=None):
        descriptor = self._publish_manifest(manifest, manifest["snapshot"], known_manifests=known_manifests)
        ptr_desc = self.client.write_json(
            self.prefix + "/" + pointer,
            {"format": POINTER_FORMAT, "manifest": descriptor},
            generation=0 if previous is None else previous["generation"],
        )
        if stats is not None:
            stats["metadata_uploaded_bytes"] += int(descriptor["size"]) + int(ptr_desc["size"])
            stats["metadata_uploaded_objects"] += 2
        if local_pointer:
            self._local_pointer(manifest, descriptor, pointer)
        return descriptor

    def _publish_manifest(self, manifest, identity, *, known_manifests=None):
        # Costs describe the one append-only physical stream, independently of
        # an algorithm cursor rollback. Every publisher retains its longest
        # committed stream and equal-prefix/cumulative tail provenance before replacing a pointer.
        if known_manifests is None:
            known_manifests = [self.current(pointer, optional=True)[0] for pointer in ("CURRENT", "FINAL", "LOGS")]
        costs = _select_costs_journal(manifest, *known_manifests)
        if costs is not None:
            manifest.setdefault("journals", {})["costs.jsonl"] = costs
            manifest.get("artifacts", {}).pop("costs.jsonl", None)
        return self.client.write_json(self.prefix + "/manifests/" + identity + ".json", manifest)

    def _torch_save(self, name, value):
        import torch
        self.cell.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.cell).free >= 32 * 1024 * 1024 * 1024:
            staging_dir = self.cell / ".gcs-staging"
            staging_dir.mkdir(parents=True, exist_ok=True)
            temp_path = staging_dir / f"{uuid.uuid4().hex}.pt"
            try:
                torch.save(value, temp_path)
                return self.client.upload_file(name, temp_path)
            finally:
                temp_path.unlink(missing_ok=True)
        writer = self.client.writer(name)
        torch.save(value, writer)
        return writer.finish()

    def _torch_load(self, descriptor):
        import torch
        with self.client.reader(descriptor) as reader:
            return torch.load(reader, map_location="cpu", weights_only=False)

    def _collect_state(self, snapshot, *, current=None, final=None, logs=None):
        """Upload only changed static artifacts and incremental journal tail slices."""
        from .checkpoint import _sync_file
        splits = self.config["evaluation_splits"][self.config["benchmark"]]
        static_names = ["config.json", "official_tasks.json", "training_summary.json", "cell_summary.json"]
        journal_names = ["costs.jsonl", "training.jsonl"]
        for split in splits:
            static_names.extend((f"{split}/evaluation_summary.json", f"{split}/evaluation_records.json"))
            journal_names.append(f"evaluation-{split}.jsonl")
        if not all((self.cell / name).is_file() for name in ("config.json", "official_tasks.json", "costs.jsonl")):
            raise ValueError("GCS snapshots require the original config, official task catalog and physical-cost journal")

        local_state = self._read_local_journal_state()
        next_local_state = {
            "artifacts": dict(local_state.get("artifacts", {})),
            "journals": dict(local_state.get("journals", {})),
            "abandoned_costs_tail_bytes": int(local_state.get("abandoned_costs_tail_bytes", 0)),
        }
        known_manifests = [m for m in (final, current, logs) if m is not None]
        valid_artifact_descs = {}
        for m in known_manifests:
            for k, desc in m.get("artifacts", {}).items():
                valid_artifact_descs.setdefault(k, []).append(desc)

        stats = _empty_transfer_stats()
        artifacts = {}
        for name in sorted(static_names):
            path = self.cell / name
            if not path.is_file():
                continue
            _sync_file(path)
            st = path.stat()
            reused_desc = None
            cached = local_state.get("artifacts", {}).get(name)
            if (cached is not None and cached.get("size") == st.st_size
                    and cached.get("mtime_ns") == st.st_mtime_ns
                    and cached.get("descriptor") in valid_artifact_descs.get(name, ())):
                reused_desc = cached["descriptor"]
            elif name == "official_tasks.json":
                for cand in valid_artifact_descs.get(name, ()):
                    if cand["size"] == st.st_size:
                        reused_desc = cand
                        break
            if reused_desc is not None:
                artifacts[name] = reused_desc
                _add_transfer(stats, "artifacts", reused_desc, uploaded=False)
                next_local_state["artifacts"][name] = {
                    "size": st.st_size, "mtime_ns": st.st_mtime_ns, "descriptor": reused_desc,
                }
            else:
                desc = self.client.upload_file(self.prefix + f"/objects/{snapshot}/artifacts/{name}", path)
                artifacts[name] = desc
                _add_transfer(stats, "artifacts", desc, uploaded=True)
                next_local_state["artifacts"][name] = {
                    "size": st.st_size, "mtime_ns": st.st_mtime_ns, "descriptor": desc,
                }

        journals = {}
        for name in sorted(journal_names):
            path = self.cell / name
            if not path.is_file():
                continue
            _sync_file(path)
            if name == "costs.jsonl":
                # A recovery truncates the old tail before handing the file
                # to a new ledger session. Bind its byte observation to one
                # stable writer epoch even if publication overlaps that handoff.
                while True:
                    before_tail_state = _read_cost_tail_state(path)
                    file_size = path.stat().st_size
                    local_committed = _committed_jsonl_size(path)
                    tail_state = _read_cost_tail_state(path)
                    if before_tail_state["writer_session_id"] == tail_state["writer_session_id"]:
                        break
            else:
                file_size = path.stat().st_size
                local_committed = _committed_jsonl_size(path)
            uncommitted_tail = max(0, file_size - local_committed)
            extra_journal_meta = {}
            if name == "costs.jsonl":
                base_journal = _select_costs_journal(*known_manifests)
                prior_discarded = 0 if base_journal is None else int(base_journal.get("discarded_incomplete_tail_bytes", 0))
                if base_journal is not None and base_journal["committed_bytes"] > local_committed:
                    raise ValueError("Existing physical-cost journal is shorter than committed remote journal")
                lineage_id = "costs-monotonic"
                consumed = _merge_tail_observations(
                    {} if base_journal is None else base_journal.get("consumed_tail_observations", {}),
                    tail_state["consumed_tail_observations"])
                discarded_total = max(next_local_state["abandoned_costs_tail_bytes"], prior_discarded,
                                      tail_state["abandoned_costs_tail_bytes"])
                next_local_state["abandoned_costs_tail_bytes"] = discarded_total
                extra_journal_meta = {
                    "uncommitted_tail_bytes_at_publish": uncommitted_tail,
                    "discarded_incomplete_tail_bytes": discarded_total,
                    "consumed_tail_observations": consumed,
                    "pending_tail_observation": (
                        {"id": _cost_tail_id(tail_state["writer_session_id"], local_committed),
                         "bytes": uncommitted_tail} if uncommitted_tail else None),
                }
            elif name == "training.jsonl":
                local_entry = local_state.get("journals", {}).get(name, {})
                local_lineage = local_entry.get("lineage_id")
                logs_j = _manifest_journal(logs, name)
                anchor_j = _manifest_journal(final if final is not None else current, name)
                if (logs_j is not None and local_lineage and logs_j.get("lineage_id") == local_lineage
                        and logs_j["committed_bytes"] <= local_committed):
                    base_journal, lineage_id = logs_j, local_lineage
                elif anchor_j is not None and anchor_j["committed_bytes"] <= local_committed:
                    base_journal = anchor_j
                    lineage_id = local_lineage or uuid.uuid4().hex
                else:
                    base_journal = None
                    lineage_id = uuid.uuid4().hex
            else:
                # evaluation-<split>.jsonl: strictly monotonic within the same active model/source snapshot.
                local_entry = local_state.get("journals", {}).get(name, {})
                local_lineage = local_entry.get("lineage_id")
                logs_j = _manifest_journal(logs, name)
                anchor_j = _manifest_journal(final if final is not None else current, name)
                logs_src = None if logs is None else logs.get("source_snapshot", logs.get("metadata", {}).get("source_snapshot"))
                same_source = current is not None and logs_src == current["snapshot"]
                if (logs_j is not None and ((local_lineage and logs_j.get("lineage_id") == local_lineage)
                        or (not local_lineage and same_source)) and logs_j["committed_bytes"] <= local_committed):
                    base_journal = logs_j
                    lineage_id = local_lineage or logs_j.get("lineage_id") or uuid.uuid4().hex
                elif anchor_j is not None and anchor_j["committed_bytes"] <= local_committed:
                    base_journal = anchor_j
                    lineage_id = local_lineage or uuid.uuid4().hex
                else:
                    base_journal = None
                    lineage_id = uuid.uuid4().hex

            base_bytes = 0 if base_journal is None else int(base_journal["committed_bytes"])
            base_segments = [] if base_journal is None else list(base_journal["segments"])
            if local_committed == 0:
                segments = []
            elif local_committed == base_bytes:
                segments = base_segments
                _add_transfer(stats, "journals", segments, uploaded=False)
            else:
                slices = [(int(seg["offset"]), int(seg["size"])) for seg in base_segments]
                slices.append((base_bytes, local_committed - base_bytes))
                while len(slices) >= 2 and slices[-2][1] < max(_JOURNAL_COALESCE_FLOOR, 2 * slices[-1][1]):
                    off_a, sz_a = slices[-2]
                    _, sz_b = slices[-1]
                    slices[-2:] = [(off_a, sz_a + sz_b)]
                reused_count = len(slices) - 1
                reused = base_segments[:reused_count]
                tail_offset, tail_size = slices[-1]
                tail_desc = self.client.upload_range(
                    self.prefix + f"/objects/{snapshot}/journals/{name}-{tail_offset}-{tail_offset + tail_size}",
                    path, offset=tail_offset, size=tail_size,
                )
                tail_seg = {"offset": tail_offset, "created_snapshot": snapshot, **tail_desc}
                segments = [*reused, tail_seg]
                _add_transfer(stats, "journals", reused, uploaded=False)
                _add_transfer(stats, "journals", tail_seg, uploaded=True)
            journals[name] = {
                "format": JOURNAL_FORMAT,
                "committed_bytes": local_committed,
                "lineage_id": lineage_id,
                "segments": segments,
                **extra_journal_meta,
            }
            next_local_state["journals"][name] = {"lineage_id": lineage_id, "committed_bytes": local_committed}
        return artifacts, journals, stats, next_local_state

    def _actor_upload(self, snapshot, directory):
        result = {}
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                name = path.relative_to(directory).as_posix()
                result[name] = self.client.upload_file(self.prefix + f"/objects/{snapshot}/model/{name}", path)
        if not result:
            raise ValueError("Actor save_pretrained directory is empty")
        return result

    def materialize_actor(self, manifest, *, target_dir=None):
        with self.client.operation():
            return self._materialize_actor(manifest, target_dir=target_dir)

    def _materialize_actor(self, manifest, *, target_dir=None):
        if target_dir is not None:
            directory = Path(target_dir)
            directory.mkdir(parents=True, exist_ok=True)
            for name, descriptor in manifest["actor"].items():
                self.client.download_file(descriptor, directory / _relative(name))
            return directory
        directory = self.cell / "actor-cache" / "model"
        marker = directory.parent / "actor.json"
        if marker.is_file() and directory.is_dir() and json.loads(marker.read_text()) == manifest["actor"]:
            if all((directory / _relative(name)).is_file() and (directory / name).stat().st_size == value["size"]
                   for name, value in manifest["actor"].items()):
                return directory
        # Remote CURRENT remains recoverable while replacing the one actor cache.
        marker.unlink(missing_ok=True)
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, descriptor in manifest["actor"].items():
            self.client.download_file(descriptor, directory / _relative(name))
        self._cache_marker(manifest["actor"])
        return directory

    def _cache_marker(self, actor):
        marker = self.cell / "actor-cache" / "actor.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        pending = marker.with_suffix(".incoming")
        pending.write_bytes(_json_bytes(actor))
        os.replace(pending, marker)

    def _base_metadata(self, manifest, actor):
        # CausalPolicy preserves base_model_path identity and reads tokenizer /
        # AutoConfig there even when weights come from checkpoint=actor.
        base = Path(self.identity["model_path"])
        base.mkdir(parents=True, exist_ok=True)
        for name in manifest["actor"]:
            if name.endswith((".safetensors", ".bin")) or name.endswith(".index.json"):
                continue
            target = base / _relative(name)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(actor / name, target)

    def restore_artifacts(self, manifest, *, logs, target_dir=None, prefer_logs_training=False, overwrite=False):
        """Restore members while the caller holds the selected manifest/LOGS pins.

        logs is required: None means confirmed absence in that same snapshot.
        """
        dest = self.cell if target_dir is None else Path(target_dir)
        dest.mkdir(parents=True, exist_ok=True)
        splits = self.config["evaluation_splits"][self.config["benchmark"]]
        known_journal_names = {"costs.jsonl", "training.jsonl", *(f"evaluation-{s}.jsonl" for s in splits)}

        valid_logs_sources = {manifest["snapshot"]}
        if isinstance(manifest.get("metadata"), dict) and manifest["metadata"].get("source_full_snapshot"):
            valid_logs_sources.add(manifest["metadata"]["source_full_snapshot"])

        static_candidates = {k: v for k, v in manifest.get("artifacts", {}).items() if k not in known_journal_names}
        if logs is not None and not manifest.get("metadata", {}).get("terminal_export_completed", False):
            logs_src = logs.get("source_snapshot", logs.get("metadata", {}).get("source_snapshot"))
            if logs_src in valid_logs_sources:
                for k, item in logs.get("artifacts", {}).items():
                    if k not in known_journal_names:
                        static_candidates[k] = item

        journal_candidates = {}
        for name in known_journal_names:
            if name == "costs.jsonl":
                continue
            j = _manifest_journal(manifest, name)
            if j is not None:
                journal_candidates[name] = j
        selected_costs = _select_costs_journal(manifest, logs)
        if selected_costs is not None:
            journal_candidates["costs.jsonl"] = selected_costs
        if logs is not None:
            logs_src = logs.get("source_snapshot", logs.get("metadata", {}).get("source_snapshot"))
            for name in known_journal_names:
                if name == "costs.jsonl":
                    continue
                lj = _manifest_journal(logs, name)
                if lj is None:
                    continue
                if name == "training.jsonl":
                    if prefer_logs_training and logs_src in valid_logs_sources:
                        if name not in journal_candidates or lj["committed_bytes"] > journal_candidates[name]["committed_bytes"]:
                            journal_candidates[name] = lj
                elif logs_src in valid_logs_sources:
                    if name not in journal_candidates or lj["committed_bytes"] > journal_candidates[name]["committed_bytes"]:
                        journal_candidates[name] = lj

        local_state = self._read_local_journal_state() if target_dir is None else None
        for name, descriptor in static_candidates.items():
            path = dest / _relative(name)
            if overwrite or not path.exists():
                self.client.download_file(descriptor, path)
                if local_state is not None:
                    st = path.stat()
                    local_state["artifacts"][name] = {
                        "size": st.st_size, "mtime_ns": st.st_mtime_ns, "descriptor": descriptor,
                    }

        for name, journal in journal_candidates.items():
            path = dest / _relative(name)
            # Local physical work after CURRENT survives. A separate LOGS
            # publication may contain more actual events after a forced exit.
            if name == "costs.jsonl" and not overwrite and path.exists():
                if path.stat().st_size >= journal["committed_bytes"]:
                    _restore_cost_tail_provenance(
                        path, journal, abandon_pending=(path.stat().st_size == journal["committed_bytes"]))
                    if local_state is not None:
                        prior_discarded = int(journal.get("discarded_incomplete_tail_bytes", 0))
                        if prior_discarded > int(local_state.get("abandoned_costs_tail_bytes", 0)):
                            local_state["abandoned_costs_tail_bytes"] = prior_discarded
                    continue
                raise ValueError("Existing physical-cost journal is shorter than remote journal; restore into the original empty cell path")
            if overwrite or not path.exists():
                self.client.download_segments(journal["segments"], path, expected_size=journal["committed_bytes"])
                if name == "costs.jsonl":
                    _restore_cost_tail_provenance(path, journal, abandon_pending=True, clear_writer=True)
                    if local_state is not None:
                        local_state["abandoned_costs_tail_bytes"] = max(
                            local_state.get("abandoned_costs_tail_bytes", 0),
                            _read_cost_tail_state(path)["abandoned_costs_tail_bytes"])

        if local_state is not None:
            self._write_local_journal_state(local_state)

    def _prepare_resume_pinned(self, pinned):
        manifest, _, _ = pinned["CURRENT"]
        logs, _, _ = pinned["LOGS"]
        if manifest["metadata"]["format"] != "fgpo-exact-checkpoint-v1":
            raise ValueError("Completed GCS model cannot resume training")
        self.restore_artifacts(manifest, logs=logs)
        self._reset_training_lineage(manifest)
        actor = self.materialize_actor(manifest)
        self._base_metadata(manifest, actor)
        return manifest, actor

    def prepare_resume(self):
        with self.transaction(), self._reader_pins(("CURRENT", "LOGS"), required=("CURRENT",), verify_cell_root=True) as pinned:
            return self._prepare_resume_pinned(pinned)

    def restore_cell(self, *, require_resume=None, optional=False):
        """Restore data under pin protection before CostLedger and CPU reservation."""
        started = time.monotonic()
        # Optional discovery must not create a local cell or its lock: the
        # runner still owns fresh initialization when no remote state exists.
        with self.open_snapshot(verify_cell_root=True) as pinned:
            final, _, final_manifest_desc = pinned["FINAL"]
            source, _, source_manifest_desc = pinned["CURRENT"]
            logs, _, _ = pinned["LOGS"]
            if source is None and final is None:
                if optional:
                    return None
                raise StorageError("restore_cell", 404)
            if require_resume is False:
                raise FileExistsError("Existing authoritative GCS cell requires explicit --resume")
            with self.transaction():
                manifest = (final if source is None or (final is not None and
                            "cell_summary.json" in final.get("artifacts", {})) else source)
                self.restore_artifacts(manifest, logs=logs)
                if (self.cell / "cell_summary.json").is_file() and final is not None:
                    self._local_pointer(final, final_manifest_desc, "FINAL")
                    return {
                        "status": "complete",
                        "snapshot": final["snapshot"],
                        "terminal_export_completed": bool(final["metadata"].get("terminal_export_completed", False)),
                        "seconds": time.monotonic() - started,
                    }
                if source is None:
                    raise ValueError("Completed GCS state lacks its durable completion summary")
                self._reset_training_lineage(source)
                self._local_pointer(source, source_manifest_desc)
                actor = self.materialize_actor(source)
                self._base_metadata(source, actor)
                return {"status": "resume", "snapshot": source["snapshot"], "actor_cache": str(actor),
                        "seconds": time.monotonic() - started}

    def restore_final(self):
        """Restore FINAL deployment model and journals under reader pin protection."""
        with self.transaction(), self._reader_pins(("FINAL", "LOGS"), required=("FINAL",), verify_cell_root=True) as pinned:
            manifest, _, manifest_desc = pinned["FINAL"]
            logs, _, _ = pinned["LOGS"]
            self.restore_artifacts(manifest, logs=logs)
            path = self.materialize_actor(manifest)
            self._base_metadata(manifest, path)
            self._local_pointer(manifest, manifest_desc, "FINAL")
            return {"snapshot": manifest["snapshot"], "model_path": str(path), "training_resume_supported": False}

    def save(self, *, policy, optimizer, reference, teacher, trainer_state, ledger_state, rng_state, model_version):
        from .checkpoint import FORMAT
        from .resources import _tensor_inventory
        started = time.monotonic()
        with self.transaction():
            previous, previous_pointer = self.current(optional=True)
            final, _ = self.current("FINAL", optional=True)
            logs, _ = self.current("LOGS", optional=True)
            if final is not None and (not trainer_state["state"].completed or
                    final["metadata"]["model_version"] != model_version or previous is None):
                raise ValueError("Final GCS artifacts already exist; this cell cannot restart training")
            if previous is not None and previous["metadata"]["format"] != FORMAT:
                raise ValueError("A completed GCS cell cannot restart training")
            snapshot = uuid.uuid4().hex
            prefix = self.prefix + "/objects/" + snapshot + "/"
            total_stats = _empty_transfer_stats()
            same = previous is not None and previous["metadata"]["model_version"] == model_version
            if same:
                actor, inventory = previous["actor"], previous["inventory"]
                _add_transfer(total_stats, "actor", list(actor.values()), uploaded=False)
            else:
                directory = self.cell / "actor-cache" / "model"
                (directory.parent / "actor.json").unlink(missing_ok=True)
                if directory.exists():
                    shutil.rmtree(directory)
                policy.save_model(directory)
                actor = self._actor_upload(snapshot, directory)
                inventory = _tensor_inventory(directory)
                self._cache_marker(actor)
                _add_transfer(total_stats, "actor", list(actor.values()), uploaded=True)
            tensors = {}
            if optimizer is not None:
                if same:
                    tensors["optimizer.pt"] = previous["tensors"]["optimizer.pt"]
                    _add_transfer(total_stats, "optimizer", tensors["optimizer.pt"], uploaded=False)
                else:
                    tensors["optimizer.pt"] = self._torch_save(prefix + "optimizer.pt", optimizer.state_dict())
                    _add_transfer(total_stats, "optimizer", tensors["optimizer.pt"], uploaded=True)
                tensors["gradients.pt"] = self._torch_save(prefix + "gradients.pt", optimizer.gradient_state())
                _add_transfer(total_stats, "gradients", tensors["gradients.pt"], uploaded=True)
            if reference is not None:
                if previous is not None:
                    tensors["reference.pt"] = previous["tensors"]["reference.pt"]
                    _add_transfer(total_stats, "reference", tensors["reference.pt"], uploaded=False)
                else:
                    tensors["reference.pt"] = self._torch_save(prefix + "reference.pt", reference.weights)
                    _add_transfer(total_stats, "reference", tensors["reference.pt"], uploaded=True)
            if teacher is not None:
                if same:
                    tensors["teacher.pt"] = previous["tensors"]["teacher.pt"]
                    _add_transfer(total_stats, "teacher", tensors["teacher.pt"], uploaded=False)
                else:
                    tensors["teacher.pt"] = self._torch_save(prefix + "teacher.pt", teacher.weights)
                    _add_transfer(total_stats, "teacher", tensors["teacher.pt"], uploaded=True)
            policy.record("checkpoint", "checkpoint.serialize_parameters", seconds=time.monotonic() - started,
                          details={"model_version": model_version, "reused_parameter_files": same, "backend": "gcs",
                                   "bulk_transfer": {k: dict(total_stats["by_category"][k])
                                                     for k in ("actor", "optimizer", "gradients", "reference", "teacher")}})
            tensors["state.pt"] = self._torch_save(prefix + "state.pt", {"trainer": trainer_state,
                "ledger": {**ledger_state, **policy.ledger.state_dict()}, "rng": rng_state})
            _add_transfer(total_stats, "state", tensors["state.pt"], uploaded=True)
            metadata = {"format": FORMAT, "model_version": model_version, "base_model_path": policy.model_path,
                "parameters_synced": True, "adaptation": policy.adaptation, "model": policy.metadata,
                "has_optimizer": optimizer is not None, "has_reference": reference is not None, "has_teacher": teacher is not None}
            artifacts, journals, state_stats, pending_local_state = self._collect_state(
                snapshot, current=previous, final=final, logs=logs)
            _merge_transfer_stats(total_stats, state_stats)
            manifest = self._new_manifest(snapshot, metadata, actor, tensors, artifacts, inventory, journals=journals)
            self._publish(manifest, previous_pointer, known_manifests=(previous, final, logs), stats=total_stats)
            self._write_local_journal_state(pending_local_state)
            deleted = self._gc()
            total_stats["deleted_objects"] = deleted
            self.last_transfer_stats = total_stats
            if self.config.get("checkpoint_gcs_release_base_weights", False):
                self.release_base_weights()
            return self.root

    def release_base_weights(self):
        """Only the declared cold-asset weights; keep tokenizer/config identity."""
        base = Path(self.identity["model_path"])
        index = base / "model.safetensors.index.json"
        names = (set(json.loads(index.read_text())["weight_map"].values()) if index.is_file()
                 else {"model.safetensors"})
        for name in names:
            path = base / _relative(name)
            if path.suffix != ".safetensors" or path.is_symlink():
                raise ValueError("Only direct original safetensors assets may be released")
            path.unlink(missing_ok=True)

    def load(self):
        with self.transaction(), self._reader_pins(("CURRENT", "LOGS"), required=("CURRENT",), verify_cell_root=True) as pinned:
            manifest, actor = self._prepare_resume_pinned(pinned)
            state = self._torch_load(manifest["tensors"]["state.pt"])
            state.update(metadata=manifest["metadata"], model_path=str(actor))
            for key in ("optimizer", "gradients", "reference", "teacher"):
                name = key + ".pt"
                if name in manifest["tensors"]:
                    state[key] = self._torch_load(manifest["tensors"][name])
            return state

    def load_final_state(self):
        """Load FINAL's state.pt under reader pin protection for interrupted finalization recovery."""
        with self.open_snapshot(("FINAL",), required=("FINAL",), verify_cell_root=True) as pinned:
            manifest, _, _ = pinned["FINAL"]
            return self._torch_load(manifest["tensors"]["state.pt"])

    def import_local_snapshot(self, *, activate_local_pointer=False):
        from .checkpoint import snapshot_directory
        from .resources import _tensor_inventory
        # Do not deserialize Adam: transfer the exact existing file bytes.
        with self.transaction():
            source, metadata = snapshot_directory(self.root)
            prior, previous = self.current(optional=True)
            final, _ = self.current("FINAL", optional=True)
            logs, _ = self.current("LOGS", optional=True)
            if final is not None:
                raise ValueError("A completed GCS cell cannot import training state")
            if metadata["base_model_path"] != self.identity["model_path"]:
                raise ValueError("Imported snapshot model_path differs from this cell")
            required = {"state.pt"}
            if metadata["has_optimizer"]:
                required.update(("optimizer.pt", "gradients.pt"))
            if metadata["has_reference"]:
                required.add("reference.pt")
            if metadata["has_teacher"]:
                required.add("teacher.pt")
            if not all((source / name).is_file() for name in required):
                raise ValueError("Local snapshot is missing a required exact-resume member")
            # state.pt is small training/RNG data; read only this file to bind
            # seed/config identity and the original cost-journal path.
            import torch
            from .protocol import require_training_identity
            state = torch.load(source / "state.pt", map_location="cpu", weights_only=False)
            require_training_identity(state["trainer"]["config"], self.config)
            if state["ledger"]["journal_path"] != str((self.cell / "costs.jsonl").resolve()):
                raise ValueError("Import requires the original physical-cost journal path")
            snapshot = uuid.uuid4().hex
            total_stats = _empty_transfer_stats()
            same = prior is not None and prior["metadata"]["model_version"] == metadata["model_version"]
            if same:
                actor = prior["actor"]
                _add_transfer(total_stats, "actor", list(actor.values()), uploaded=False)
            else:
                actor = self._actor_upload(snapshot, source / "model")
                _add_transfer(total_stats, "actor", list(actor.values()), uploaded=True)
            tensors = {}
            tensor_categories = {
                "optimizer.pt": "optimizer",
                "gradients.pt": "gradients",
                "reference.pt": "reference",
                "teacher.pt": "teacher",
                "state.pt": "state",
            }
            for name in required:
                reuse = prior is not None and (name == "reference.pt" or (same and name in {"optimizer.pt", "teacher.pt"}))
                if reuse:
                    tensors[name] = prior["tensors"][name]
                    _add_transfer(total_stats, tensor_categories[name], tensors[name], uploaded=False)
                else:
                    tensors[name] = self.client.upload_file(self.prefix + f"/objects/{snapshot}/{name}", source / name)
                    _add_transfer(total_stats, tensor_categories[name], tensors[name], uploaded=True)
            artifacts, journals, state_stats, pending_local_state = self._collect_state(
                snapshot, current=prior, final=final, logs=logs)
            _merge_transfer_stats(total_stats, state_stats)
            manifest = self._new_manifest(snapshot, metadata, actor, tensors, artifacts,
                                          _tensor_inventory(source / "model"), journals=journals)
            manifest["source_local_snapshot"] = source.name
            self._publish(manifest, previous, local_pointer=activate_local_pointer,
                          known_manifests=(prior, final, logs), stats=total_stats)
            self._write_local_journal_state(pending_local_state)
            deleted = self._gc()
            total_stats["deleted_objects"] = deleted
            self.last_transfer_stats = total_stats
            return {"snapshot": snapshot, "source_local_snapshot": source.name, "uri": "gs://" + self.client.bucket + "/" + self.prefix,
                    "model_version": metadata["model_version"], "round": state["trainer"]["state"].round,
                    "phase": state["trainer"]["state"].phase,
                    "local_current_preserved": not activate_local_pointer, "transfer": total_stats}

    def publish_logs(self):
        with self.transaction():
            final, _, final_manifest_desc = self._current_entry("FINAL", optional=True)
            current, _, current_manifest_desc = self._current_entry("CURRENT", optional=True)
            source = final if final is not None else current
            if source is None:
                raise StorageError("publish_logs", 404)
            logs, previous_logs_pointer, previous_logs_manifest_desc = self._current_entry("LOGS", optional=True)
            snapshot = uuid.uuid4().hex
            artifacts, journals, stats, pending_local_state = self._collect_state(
                snapshot, current=current, final=final, logs=logs)
            manifest = self._new_manifest(
                snapshot, {"format": LOGS_METADATA_FORMAT, "source_snapshot": source["snapshot"]},
                {}, {}, artifacts, {}, journals=journals)
            manifest["source_snapshot"] = source["snapshot"]
            new_logs_manifest_desc = self._publish(
                manifest, previous_logs_pointer, pointer="LOGS", local_pointer=False,
                known_manifests=(current, final, logs), stats=stats)
            self._write_local_journal_state(pending_local_state)
            deleted = self._gc_logs_superseded(
                old_logs=(logs, previous_logs_manifest_desc),
                keep=((current, current_manifest_desc), (final, final_manifest_desc), (manifest, new_logs_manifest_desc)),
            )
            stats = {**stats, "deleted_objects": deleted}
            self.last_transfer_stats = stats
            return {
                "snapshot": snapshot,
                "manifest_format": MANIFEST_FORMAT,
                "artifacts": sorted(manifest["artifacts"]),
                "journals": {name: {"committed_bytes": item["committed_bytes"], "segments": len(item["segments"])}
                             for name, item in manifest["journals"].items()},
                "transfer": stats,
            }

    def _gc_logs_superseded(self, *, old_logs, keep):
        """O(1) per-publication reclamation for superseded LOGS objects without listing.

        Defers deletion when any concurrent reader pin is active so pinned manifests
        and journal segments remain available until full GC after reader release.
        """
        old_manifest, old_desc = old_logs
        if old_manifest is None and old_desc is None:
            return 0
        if self._active_reader_pins():
            return 0
        protected = set()
        for m, m_desc in keep:
            protected.update(_manifest_descriptors(m, m_desc))
        candidates = _manifest_descriptors(old_manifest, old_desc)
        deleted = 0
        for name, generation in sorted(candidates - protected):
            self.client.delete({"name": name, "generation": generation})
            deleted += 1
        return deleted

    def prepare_completed(self, *, provenance):
        from .checkpoint import COMPLETED_FORMAT
        with self.transaction():
            source, _ = self.current()
            previous, previous_pointer = self.current("FINAL", optional=True)
            logs, _ = self.current("LOGS", optional=True)
            snapshot = uuid.uuid4().hex
            total_stats = _empty_transfer_stats()
            _add_transfer(total_stats, "actor", list(source["actor"].values()), uploaded=False)
            metadata = {**source["metadata"], "format": COMPLETED_FORMAT, "cell_root": str(self.cell.resolve()),
                "training_resume_supported": False, "has_optimizer": False, "has_reference": False, "has_teacher": False,
                "source_full_snapshot": source["snapshot"], "terminal_export_completed": False,
                "retain_training_artifacts": bool(provenance.get("retain_training_artifacts", True))}
            tensors = {key: value for key, value in source["tensors"].items() if key in {"state.pt", "teacher.pt"}}
            for key, desc in tensors.items():
                _add_transfer(total_stats, "state" if key == "state.pt" else "teacher", desc, uploaded=False)
            artifacts, journals, state_stats, pending_local_state = self._collect_state(
                snapshot, current=source, final=previous, logs=logs)
            _merge_transfer_stats(total_stats, state_stats)
            prov_desc = self.client.write_json(self.prefix + f"/objects/{snapshot}/provenance.json", provenance)
            artifacts["provenance.json"] = prov_desc
            _add_transfer(total_stats, "artifacts", prov_desc, uploaded=True)
            final = self._new_manifest(snapshot, metadata, source["actor"], tensors, artifacts,
                                       source["inventory"], journals=journals)
            self._publish(final, previous_pointer, pointer="FINAL",
                          known_manifests=(source, previous, logs), stats=total_stats)
            self._write_local_journal_state(pending_local_state)
            self.last_transfer_stats = total_stats
            return {"path": str((self.cell / "final_checkpoint").resolve()), "format": COMPLETED_FORMAT, "backend": "gcs",
                "snapshot": snapshot, "source_full_snapshot": source["snapshot"], "training_resume_supported": False,
                "retain_training_artifacts": metadata["retain_training_artifacts"], "gcs_uri": "gs://" + self.client.bucket + "/" + self.prefix,
                "transfer": total_stats}

    def reclaim_completed(self, summary_path, *, keep_journal_state=False):
        from .checkpoint import _reclaim_training_artifacts
        with self.transaction():
            summary_path = Path(summary_path)
            if summary_path.resolve() != (self.cell / "cell_summary.json").resolve():
                raise ValueError("Completion summary must belong to the original cell")
            summary = json.loads(summary_path.read_text())
            final, final_pointer = self.current("FINAL")
            artifact = summary.get("completed_checkpoint", {})
            if (summary.get("status") != "complete" or artifact.get("snapshot") != final["snapshot"]
                    or artifact.get("source_full_snapshot") != final["metadata"]["source_full_snapshot"]):
                raise ValueError("Remote final model and durable completed cell summary differ")
            source, pointer = self.current(optional=True)
            logs, _ = self.current("LOGS", optional=True)
            new_manifest_id = uuid.uuid4().hex
            total_stats = _empty_transfer_stats()
            _add_transfer(total_stats, "actor", list(final["actor"].values()), uploaded=False)
            for key, desc in final.get("tensors", {}).items():
                _add_transfer(total_stats, "state" if key == "state.pt" else "teacher", desc, uploaded=False)
            artifacts, journals, state_stats, pending_local_state = self._collect_state(
                new_manifest_id, current=source, final=final, logs=logs)
            _merge_transfer_stats(total_stats, state_stats)
            # Persist official summaries/records, diagnostics and physical costs
            # before deleting any exact-resume object. FINAL identity stays fixed.
            final = dict(final)
            final["format"] = MANIFEST_FORMAT
            merged_artifacts = {k: v for k, v in final.get("artifacts", {}).items() if k not in journals}
            merged_artifacts.update(artifacts)
            final["artifacts"] = merged_artifacts
            final["journals"] = journals
            descriptor = self._publish_manifest(final, new_manifest_id, known_manifests=(source, final, logs))
            ptr_desc = self.client.write_json(
                self.prefix + "/FINAL", {"format": POINTER_FORMAT, "manifest": descriptor},
                generation=final_pointer["generation"],
            )
            total_stats["metadata_uploaded_bytes"] += int(descriptor["size"]) + int(ptr_desc["size"])
            total_stats["metadata_uploaded_objects"] += 2
            self._local_pointer(final, descriptor, "FINAL")
            self._write_local_journal_state(pending_local_state)
            if source is not None:
                if source["snapshot"] != artifact["source_full_snapshot"]:
                    raise ValueError("Training CURRENT changed after final export")
                self.client.delete(pointer)
            deleted = self._gc()
            self.last_transfer_stats = {**total_stats, "deleted_objects": deleted}
            local = self.cell / "checkpoint"
            if local.exists():
                if local.is_symlink():
                    raise ValueError("Refuse linked local checkpoint cleanup")
                # Migration never silently deletes legacy large local snapshots.
                if (local / "snapshots").exists():
                    raise ValueError("Explicit operator migration must reclaim local legacy snapshots before finalization")
                shutil.rmtree(local)
            if not artifact.get("retain_training_artifacts", True):
                _reclaim_training_artifacts(self.cell)
            cache = self.cell / "actor-cache"
            if cache.exists():
                if cache.is_symlink():
                    raise ValueError("Refuse linked actor-cache cleanup")
                shutil.rmtree(cache)
            if not keep_journal_state:
                self._journal_state_path().unlink(missing_ok=True)
            return source is not None

    def publish_final_artifacts(self, summary_path):
        """Publish finalized cell_summary.json and costs.jsonl to FINAL and LOGS."""
        with self.transaction():
            summary_path = Path(summary_path)
            if summary_path.resolve() != (self.cell / "cell_summary.json").resolve():
                raise ValueError("Completion summary must belong to the original cell")
            summary = json.loads(summary_path.read_text())
            final, final_pointer = self.current("FINAL")
            artifact = summary.get("completed_checkpoint", {})
            if (summary.get("status") != "complete"
                    or summary.get("finalization_stage") != "finalized"
                    or artifact.get("snapshot") != final["snapshot"]
                    or artifact.get("source_full_snapshot") != final["metadata"]["source_full_snapshot"]):
                raise ValueError("Remote final model and durable finalized cell summary differ")
            logs, previous_logs_pointer = self.current("LOGS", optional=True)
            manifest_id = uuid.uuid4().hex
            total_stats = _empty_transfer_stats()
            _add_transfer(total_stats, "actor", list(final["actor"].values()), uploaded=False)
            for key, desc in final.get("tensors", {}).items():
                _add_transfer(total_stats, "state" if key == "state.pt" else "teacher", desc, uploaded=False)
            artifacts, journals, state_stats, pending_local_state = self._collect_state(
                manifest_id, current=None, final=final, logs=logs)
            _merge_transfer_stats(total_stats, state_stats)

            final_updated = dict(final)
            final_updated["format"] = MANIFEST_FORMAT
            final_updated["metadata"] = {**final["metadata"], "terminal_export_completed": True}
            merged_artifacts = {k: v for k, v in final.get("artifacts", {}).items() if k not in journals}
            merged_artifacts.update(artifacts)
            final_updated["artifacts"] = merged_artifacts
            final_updated["journals"] = journals
            final_desc = self._publish_manifest(final_updated, manifest_id, known_manifests=(final, logs))
            logs_snapshot = uuid.uuid4().hex
            logs_manifest = self._new_manifest(
                logs_snapshot, {"format": LOGS_METADATA_FORMAT, "source_snapshot": final["snapshot"]},
                {}, {}, artifacts, {}, journals=final_updated["journals"],
            )
            logs_manifest["source_snapshot"] = final["snapshot"]
            self._publish(
                logs_manifest, previous_logs_pointer, pointer="LOGS", local_pointer=False,
                known_manifests=(final_updated, logs), stats=total_stats,
            )
            # FINAL's completion flag is the commit point: the matching LOGS
            # mirror must already be durable when readers observe this flag.
            final_ptr_desc = self.client.write_json(
                self.prefix + "/FINAL", {"format": POINTER_FORMAT, "manifest": final_desc},
                generation=final_pointer["generation"],
            )
            total_stats["metadata_uploaded_bytes"] += int(final_desc["size"]) + int(final_ptr_desc["size"])
            total_stats["metadata_uploaded_objects"] += 2
            self._local_pointer(final_updated, final_desc, "FINAL")
            self._write_local_journal_state(pending_local_state)
            deleted = self._gc()
            self._journal_state_path().unlink(missing_ok=True)
            total_stats["deleted_objects"] = deleted
            self.last_transfer_stats = total_stats
            return {
                "final_snapshot": final["snapshot"],
                "logs_snapshot": logs_snapshot,
                "manifest_format": MANIFEST_FORMAT,
                "journals": {name: {"committed_bytes": item["committed_bytes"], "segments": len(item["segments"])}
                             for name, item in final_updated["journals"].items()},
                "transfer": total_stats,
            }

    def status(self):
        with self._cell_lock_if_present(), self._reader_pins(("CURRENT", "FINAL", "LOGS"), verify_cell_root=False) as pinned:
            manifest, pointer, _ = pinned["CURRENT"]
            final, final_pointer, _ = pinned["FINAL"]
            logs, logs_pointer, _ = pinned["LOGS"]
        active = manifest if manifest is not None else final
        journal_source = logs if logs is not None else active
        splits = self.config["evaluation_splits"][self.config["benchmark"]]
        known_journal_names = {"costs.jsonl", "training.jsonl", *(f"evaluation-{s}.jsonl" for s in splits)}
        journals = {}
        for source_m in (active, journal_source):
            if source_m is None:
                continue
            for name in sorted(set(source_m.get("journals", {})) | (known_journal_names & set(source_m.get("artifacts", {})))):
                if name == "costs.jsonl":
                    continue
                entry = _manifest_journal(source_m, name)
                if entry is not None:
                    journals[name] = {
                        "format": entry["format"],
                        "committed_bytes": entry["committed_bytes"],
                        "segments": len(entry["segments"]),
                    }
        best_costs = _select_costs_journal(manifest, final, logs)
        if best_costs is not None:
            journals["costs.jsonl"] = {
                "format": best_costs["format"],
                "committed_bytes": best_costs["committed_bytes"],
                "segments": len(best_costs["segments"]),
                "uncommitted_tail_bytes_at_publish": int(best_costs.get("uncommitted_tail_bytes_at_publish", 0)),
                "discarded_incomplete_tail_bytes": int(best_costs.get("discarded_incomplete_tail_bytes", 0)),
            }
        return {
            "status": "complete" if final is not None and manifest is None else "resume" if manifest is not None else "absent",
            "uri": "gs://" + self.client.bucket + "/" + self.prefix,
            "manifest_format": None if active is None else active["format"],
            "snapshot": None if active is None else active["snapshot"],
            "model_version": None if active is None else active["metadata"]["model_version"],
            "terminal_export_completed": None if final is None else bool(final["metadata"].get("terminal_export_completed", False)),
            "current_generation": None if pointer is None else pointer["generation"],
            "final_generation": None if final_pointer is None else final_pointer["generation"],
            "logs_snapshot": None if logs is None else logs["snapshot"],
            "logs_generation": None if logs_pointer is None else logs_pointer["generation"],
            "artifacts": [] if active is None else sorted(active.get("artifacts", {})),
            "journals": journals,
        }

    def pull(self, dest_dir, *, with_final_model=False):
        dest = Path(dest_dir)
        with self._cell_lock_if_present(), self._reader_pins(("CURRENT", "FINAL", "LOGS"), verify_cell_root=False) as pinned:
            manifest, _, _ = pinned["CURRENT"]
            final, _, _ = pinned["FINAL"]
            logs, _, _ = pinned["LOGS"]
            active = final if final is not None else manifest
            if active is None and logs is None:
                raise StorageError("pull", 404)
            base_manifest = active if active is not None else logs
            self.restore_artifacts(base_manifest, target_dir=dest, logs=logs,
                                   prefer_logs_training=True, overwrite=True)
            final_model_dir = None
            if with_final_model and final is not None:
                final_model_dir = str(self._materialize_actor(final, target_dir=dest / "final_checkpoint" / "model"))
        journals = {}
        splits = self.config["evaluation_splits"][self.config["benchmark"]]
        for name in ("costs.jsonl", "training.jsonl", *(f"evaluation-{s}.jsonl" for s in splits)):
            path = dest / name
            if path.is_file():
                journals[name] = path.stat().st_size
        return {
            "status": "complete" if final is not None and manifest is None else "resume" if manifest is not None else "logs_only",
            "uri": "gs://" + self.client.bucket + "/" + self.prefix,
            "snapshot": base_manifest["snapshot"],
            "manifest_format": base_manifest["format"],
            "dest_dir": str(dest),
            "journals": journals,
            "final_model_dir": final_model_dir,
        }

    def _gc(self):
        live = set()
        for pointer in ("CURRENT", "FINAL", "LOGS"):
            manifest, _, manifest_descriptor = self._current_entry(pointer, optional=True)
            live.update(_manifest_descriptors(manifest, manifest_descriptor))
        for _, pin_data in self._active_reader_pins():
            for m_desc in pin_data.get("manifests", ()):
                self._check_descriptor(m_desc)
                try:
                    manifest = self._validate_manifest(self.client.read_json(m_desc), verify_cell_root=False)
                except StorageError as error:
                    if error.status == 404:
                        continue
                    raise
                live.update(_manifest_descriptors(manifest, m_desc))
        deleted = 0
        for section in ("objects/", "manifests/"):
            for item in self.client.list_objects(self.prefix + "/" + section):
                if (item["name"], item["generation"]) not in live:
                    self.client.delete(item)
                    deleted += 1
        return deleted
