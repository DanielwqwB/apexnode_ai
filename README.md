# SentryMesh VigilantPath API

FastAPI service for the trained ST-GNN hazard model.

## Local Run

```bash
pip install -r requirements.txt
uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- `http://localhost:8000/`
- `http://localhost:8000/model/info`
- `http://localhost:8000/predict/demo`

## Deploy To Render

This repo includes `render.yaml`, `requirements.txt`, and `.python-version`.

1. Push this repository to GitHub.
2. In Render, create a new Blueprint from this repo, or create a Python Web Service manually.
3. Manual Web Service settings:
   - Build command: `pip install --upgrade pip && pip install -r requirements.txt`
   - Start command: `uvicorn serve:app --host 0.0.0.0 --port $PORT`
   - Health check path: `/`
4. After deploy, use the Render URL as your Flutter API base URL:

```text
https://sentrymesh-vigilantpath-api.onrender.com
```

The exact host can differ if Render changes the service slug.

## Flutter Fetch Example

Add `http` to `pubspec.yaml`, then call the demo endpoint:

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

const apiBaseUrl = 'https://sentrymesh-vigilantpath-api.onrender.com';

Future<Map<String, dynamic>> fetchDemoPrediction() async {
  final uri = Uri.parse('$apiBaseUrl/predict/demo');
  final response = await http.get(uri);

  if (response.statusCode != 200) {
    throw Exception('API error ${response.statusCode}: ${response.body}');
  }

  return jsonDecode(response.body) as Map<String, dynamic>;
}
```

For real predictions, call `GET /model/info` first and send raw feature values in
the returned `feature_cols` order:

```dart
Future<Map<String, dynamic>> predictNodes() async {
  final uri = Uri.parse('$apiBaseUrl/predict');
  final response = await http.post(
    uri,
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({
      'nodes': [
        {
          'node_id': 42,
          'features': [
            14.5, 121.0, 6, 162, 14, 1, 8.2, 7.3, 0.3, 0.6,
            0.4, 0.2, 0.05, 65, 985, 18, 120, 20, 4
          ],
        },
      ],
    }),
  );

  if (response.statusCode != 200) {
    throw Exception('API error ${response.statusCode}: ${response.body}');
  }

  return jsonDecode(response.body) as Map<String, dynamic>;
}
```
