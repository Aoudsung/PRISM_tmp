#!/usr/bin/env bash
set -euo pipefail
umask 077
test "$(hostname -s)" = mgt01
cd -- "$(dirname -- "$0")"
test "$PWD" = /apps/users/adminuser/fgpo-verified-feedback-20261118-v5

adapter=/apps/users/adminuser/fgpo-v4-webshop-adapter-20261118.in2tA1Hp
pilot=/apps/users/adminuser/fgpo-feedback-pilot-20261118-a1
stage=$(mktemp -d -- "$PWD/.v5src.XXXXXXXX")
cp -a -- "$adapter/src/fgpo" "$stage/fgpo"
cp -- "$pilot/collect.py" "$stage/pilot_collect_v5.py"
patch --batch --forward --directory="$stage" -p1 < contract.patch
mv -T -- "$stage" "$PWD/v5src"
