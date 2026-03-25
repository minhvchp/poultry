# Poultry Disease Detector App

Ứng dụng phát hiện bệnh gia cầm qua mẫu phân, sử dụng model AI (MobileNetV3 + Knowledge Distillation).

## Thông tin Model

| Thuộc tính | Giá trị |
|---|---|
| Architecture | MobileNetV3-Small (Student KD) |
| Input size | 224 × 224 × 3 (RGB) |
| Normalization | ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) |
| Output classes | 4 |
| Output format | Logits → Softmax |

### Các lớp đầu ra

| Index | Class | Tiếng Việt |
|---|---|---|
| 0 | Coccidiosis | Bệnh Cầu Trùng |
| 1 | Healthy | Bình Thường |
| 2 | NCD | Bệnh Newcastle |
| 3 | Salmonella | Bệnh Salmonella |

---

## Cài đặt & Chạy

### Bước 1: Cài Node.js

Tải Node.js từ: https://nodejs.org/en/download/
(Chọn phiên bản LTS)

### Bước 2: Cài Expo CLI

```bash
npm install -g expo-cli
```

### Bước 3: Sao chép file model

Sao chép file `student_clean.onnx` vào thư mục `assets/`:

```
PoultryApp/
├── assets/
│   ├── student_clean.onnx   ← COPY FILE NÀY VÀO ĐÂY
│   ├── icon.png
│   └── splash.png
├── App.js
├── package.json
└── ...
```

### Bước 4: Cài thư viện JPEG decoder

Do ONNX model cần pixel data thực, cần cài thêm một trong hai:

**Option A: jpeg-js (thuần JS, dễ cài)**
```bash
npm install jpeg-js
```

**Option B: react-native-image-resizer (native, nhanh hơn)**
```bash
npx expo install react-native-image-resizer
```

### Bước 5: Cài dependencies

```bash
cd PoultryApp
npm install
```

### Bước 6: Cài Expo Go hoặc build APK

**For Testing (Expo Go):**
```bash
npx expo start --tunnel
```
Scan QR code bằng app Expo Go trên Android.

**For APK Build (Production):**
```bash
# Cài EAS CLI
npm install -g eas-cli

# Đăng nhập Expo account (tạo free tại expo.dev)
eas login

# Build APK
eas build -p android --profile preview
```

---

## Kiến trúc kỹ thuật

```
Chọn ảnh (expo-image-picker)
    ↓
Resize về 224×224 (expo-image-manipulator)
    ↓
Decode JPEG → RGBA pixels (jpeg-js)
    ↓
Normalize: (RGB / 255 - mean) / std → Float32Array
    ↓
Reshape: [1, 3, 224, 224] CHW format
    ↓
ONNX Runtime Inference (onnxruntime-react-native)
    ↓
Softmax → Probabilities
    ↓
argmax → Class prediction
    ↓
Hiển thị kết quả + recommendations
```

---

## Xử lý pixel (QUAN TRỌNG)

Để inference chính xác, cần decode JPEG đúng cách.
Cập nhật App.js để sử dụng jpeg-js:

```javascript
import jpeg from 'jpeg-js';
import * as FileSystem from 'expo-file-system';

async function decodeJpegToPixels(uri) {
  // Đọc file ảnh dưới dạng base64
  const base64 = await FileSystem.readAsStringAsync(uri, {
    encoding: FileSystem.EncodingType.Base64,
  });
  
  // Decode base64 → Uint8Array
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  
  // Decode JPEG → RGBA pixels
  const decoded = jpeg.decode(bytes, { useTArray: true });
  return decoded.data; // Uint8Array RGBA format
}
```

---

## Cấu trúc thư mục

```
PoultryApp/
├── App.js              ← Main component + UI
├── package.json        ← Dependencies
├── app.json            ← Expo config
├── babel.config.js     ← Babel config
└── assets/
    ├── student_clean.onnx  ← AI model (copy từ parent folder)
    ├── icon.png            ← App icon (tạo 1024×1024)
    ├── adaptive-icon.png   ← Android adaptive icon
    ├── splash.png          ← Splash screen
    └── favicon.png         ← Web favicon
```

---

## Tạo placeholder assets

Nếu chưa có icon/splash, tạo file trống:

```bash
# Trên Windows PowerShell, trong thư mục PoultryApp/assets:
$null > icon.png
$null > adaptive-icon.png  
$null > splash.png
$null > favicon.png
```

Hoặc dùng bất kỳ ảnh PNG nào.

---

## Lưu ý

- **onnxruntime-react-native** yêu cầu **development build** (không chạy được trên Expo Go thông thường)
- Sử dụng `eas build` để build APK có đầy đủ native modules
- Hoặc tạo **custom development client** với `expo-dev-client`
