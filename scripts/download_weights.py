#!/usr/bin/env python3
"""Download a reviewer-supplied weights URL directly and verify its SHA-256."""
import argparse
import hashlib
import os
import uuid
from pathlib import Path
import urllib.request

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--url',required=True);p.add_argument('--sha256',required=True);p.add_argument('--output',required=True)
a=p.parse_args()
if not a.url.startswith('https://'):raise ValueError('Use an HTTPS weight URL.')
if len(a.sha256)!=64 or any(c not in '0123456789abcdefABCDEF' for c in a.sha256):raise ValueError('Expected a SHA-256 hex digest.')
path=Path(a.output)
if path.exists():raise FileExistsError(path)
path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
# Per-process direct route even if proxy variables are configured in the shell.
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
h=hashlib.sha256()
try:
 with opener.open(a.url,timeout=60) as source,tmp.open('xb') as dest:
  while chunk:=source.read(8*1024*1024):dest.write(chunk);h.update(chunk)
 if h.hexdigest()!=a.sha256.lower():raise ValueError('Downloaded file SHA-256 mismatch.')
 os.link(tmp,path) # Atomic no-clobber publication of the verified local file.
 tmp.unlink()
except BaseException:
 tmp.unlink(missing_ok=True);raise
print(path)
