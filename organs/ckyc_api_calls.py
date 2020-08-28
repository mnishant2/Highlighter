import time
import base64
import os
import json
import http.client

def upload_zip_2_elastic(cif, workerid, zipfilePath, created, status):

	try:
		with open(zipfilePath, "rb") as zip_file:
			data = zip_file.read()
			base64_bytes = base64.b64encode(data)
			base64_string = base64_bytes.decode('utf-8')
	except Exception as e:
		print (e)

	try:
		import http.client
		conn = http.client.HTTPConnection("filesystem01.sbickyc.signzy.tech")
		payload = {"cif": str(cif), "workerid": "workerid",  "zipfile": base64_string , "created": created , "status": "status" }
		payload = json.dumps(payload)
		headers = {
		'content-type': "application/json",
		'cache-control': "no-cache",
		'postman-token': "8d44aaa0-b218-0451-d498-bed7d405fd5d"
		}
		conn.request("POST", "/upload.php", payload, headers)
		res = conn.getresponse()
		data = res.read()
		# print(data.decode("utf-8"))
	except Exception as e:
		print (e)

if __name__ == "__main__":
	cif = 88888888889
	workerid = "worker-0"
	zipfile = "/home/signzy-engine/abs/ckyc-worker/tmp/999.zip"
	created = int(round(time.time() * 1000))
	status = "new"
	upload_zip_2_elastic(cif, workerid, zipfile, created, status)