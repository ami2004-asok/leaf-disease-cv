import requests

url = "http://127.0.0.1:8000/predict"

image_path = "data/val/late_blight/f84a91cc-ad11-409e-9115-1af741e5726d___RS_Late.B 5166.jpg"

with open(image_path, "rb") as image:
    files = {
        "file": (
            image_path,      # filename
            image,           # file object
            "image/jpeg"     # MIME type
        )
    }

    response = requests.post(
        url,
        files=files
    )

print("Status Code:", response.status_code)
print("Response:")
print(response.json())