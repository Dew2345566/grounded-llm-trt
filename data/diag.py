import os, json, urllib.request, urllib.error

key = os.environ.get("GEMINI_API_KEY")
print("key present:", bool(key), "length:", len(key or ""))

# 1. Which models does this key actually have?
try:
    u = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    data = json.loads(urllib.request.urlopen(u, timeout=60).read())
    names = [m["name"].split("/")[-1] for m in data.get("models", [])]
    print("available:", [n for n in names if "flash" in n or "pro" in n][:10])
except urllib.error.HTTPError as e:
    print("LIST FAILED", e.code, e.read().decode()[:800])

# 2. One real call, with the body printed on failure
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}"
body = {"contents": [{"parts": [{"text": "Reply with the single word: ok"}]}]}
req = urllib.request.Request(url, data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
try:
    print("OK:", urllib.request.urlopen(req, timeout=60).read()[:300])
except urllib.error.HTTPError as e:
    print("CALL FAILED", e.code)
    print(e.read().decode()[:1500])