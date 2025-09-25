# Mobile App integration

## 1. Send an image to the API
Make a POST request with the PNG encoded image as body. The request should be done to 
{root_url}/api/food. If the POST request was succesful you get a response like this:

```http
HTTP/1.1 200 OK
Server: Werkzeug/3.0.4 Python/3.12.4
Date: Thu, 25 Sep 2025 13:25:55 GMT
Content-Type: application/json
Content-Length: 51
Connection: close

{
  "id": "c02de3f4-8b9f-468b-9ddc-82aefc0c8820"
}
```
extract the id from the json.

## 2. Get result
Make a GET request to {root_url}/food/c02de3f4-8b9f-468b-9ddc-82aefc0c8820/analysis_result to get the result. 

```http
HTTP/1.1 200 OK
Server: Werkzeug/3.0.4 Python/3.12.4
Date: Thu, 25 Sep 2025 13:35:14 GMT
Content-Type: application/json
Content-Length: 33
Connection: close

{
  "food_type": "Carrot cake"
}
```