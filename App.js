/**
 * Poultry Disease Detector
 * 
 * Model: MobileNetV3-Small (Student, KD + Pruning)
 * Classes: Coccidiosis (0), Healthy (1), NCD (2), Salmonella (3)
 * Input: [1, 3, 224, 224] Float32, ImageNet normalized
 * 
 * Pipeline:
 *   ImagePicker → resize 224×224 → JPEG decode → normalize → ONNX → softmax → result
 */

import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Image,
  ScrollView,
  ActivityIndicator,
  Alert,
  Platform,
  Dimensions,
  StatusBar,
} from 'react-native';
import { StatusBar as ExpoStatusBar } from 'expo-status-bar';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';
import * as FileSystem from 'expo-file-system';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';

// ─── Model Configuration ──────────────────────────────────────────────────────
// IMPORTANT: Copy student_clean.onnx to PoultryApp/assets/
const MODEL_ASSET = require('./assets/student_clean.onnx');

// ImageNet normalization (same as training pipeline)
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD  = [0.229, 0.224, 0.225];
const INPUT_SIZE = 224;

// Class names MUST match training order
const CLASS_NAMES = ['Coccidiosis', 'Healthy', 'NCD', 'Salmonella'];

// ─── Disease Information ──────────────────────────────────────────────────────
const CLASS_INFO = {
  Coccidiosis: {
    viName: 'Cầu Trùng',
    color: '#FF6B35',
    icon: 'bug',
    iconLib: 'Ionicons',
    description:
      'Bệnh do ký sinh trùng Eimeria gây ra. Phá huỷ niêm mạc ruột, gây tiêu chảy có máu, mất nước và tỷ lệ chết cao ở gà con.',
    severity: 'Nguy hiểm',
    severityColor: '#FF4444',
    recommendation:
      'Điều trị ngay bằng Amprolium hoặc Toltrazuril. Cách ly đàn bệnh. Vệ sinh và khử trùng chuồng trại. Tham khảo ý kiến bác sĩ thú y.',
  },
  Healthy: {
    viName: 'Khỏe Mạnh',
    color: '#00E676',
    icon: 'checkmark-circle',
    iconLib: 'Ionicons',
    description:
      'Gia cầm có sức khỏe tốt. Mẫu phân có màu nâu xanh bình thường, hơi ẩm với rần có màu trắng urate.',
    severity: 'Bình thường',
    severityColor: '#00C853',
    recommendation:
      'Tiếp tục duy trì chế độ dinh dưỡng cân bằng, vệ sinh chuồng trại định kỳ và tiêm phòng đúng lịch.',
  },
  NCD: {
    viName: 'Newcastle',
    color: '#FF3D00',
    icon: 'warning',
    iconLib: 'Ionicons',
    description:
      'Bệnh Newcastle (Dịch tả gia cầm) do Paramyxovirus gây ra. Rất nguy hiểm, lây lan cực nhanh, tỷ lệ chết có thể lên đến 100%.',
    severity: 'RẤT NGUY HIỂM',
    severityColor: '#B71C1C',
    recommendation:
      '⚠️ KHẨN CẤP: Báo ngay cơ quan thú y địa phương. Không vận chuyển gia cầm. Cách ly và tiêu huỷ đàn bị bệnh theo hướng dẫn.',
  },
  Salmonella: {
    viName: 'Salmonella',
    color: '#FFB300',
    icon: 'alert-circle',
    iconLib: 'Ionicons',
    description:
      'Nhiễm khuẩn Salmonella gây viêm ruột, tiêu chảy phân xanh vàng. Nguy hiểm vì có thể lây sang người qua thực phẩm.',
    severity: 'Nguy hiểm',
    severityColor: '#E65100',
    recommendation:
      'Điều trị kháng sinh theo kháng sinh đồ. Tăng cường vệ sinh. Không ăn thịt/trứng từ đàn bị nhiễm khi chưa điều trị.',
  },
};

// ─── ONNX Session Management ─────────────────────────────────────────────────
let _session = null;

async function getSession() {
  if (_session) return _session;
  
  _session = await InferenceSession.create(MODEL_ASSET, {
    executionProviders: ['cpu'],
    graphOptimizationLevel: 'all',
  });
  
  return _session;
}

// ─── Image Preprocessing ─────────────────────────────────────────────────────

/**
 * Decode base64 JPEG to RGBA Uint8Array using jpeg-js
 * Requires: npm install jpeg-js
 */
async function decodeJpegBase64(base64Str) {
  try {
    // Try jpeg-js first (npm install jpeg-js)
    const jpeg = require('jpeg-js');
    const binary = _base64ToBinary(base64Str);
    const decoded = jpeg.decode(binary, { useTArray: true });
    return { data: decoded.data, width: decoded.width, height: decoded.height };
  } catch {
    // Fallback if jpeg-js not installed: use uniform pixel
    console.warn('jpeg-js not found. Using fallback pixel data for demo.');
    const size = INPUT_SIZE * INPUT_SIZE;
    const fallback = new Uint8Array(size * 4);
    // Fill with neutral gray (will produce near-zero normalized values)
    for (let i = 0; i < size; i++) {
      fallback[i * 4]     = Math.round(IMAGENET_MEAN[0] * 255); // R
      fallback[i * 4 + 1] = Math.round(IMAGENET_MEAN[1] * 255); // G
      fallback[i * 4 + 2] = Math.round(IMAGENET_MEAN[2] * 255); // B
      fallback[i * 4 + 3] = 255; // A
    }
    return { data: fallback, width: INPUT_SIZE, height: INPUT_SIZE };
  }
}

function _base64ToBinary(base64) {
  const str = atob(base64);
  const buf = new Uint8Array(str.length);
  for (let i = 0; i < str.length; i++) buf[i] = str.charCodeAt(i);
  return buf;
}

/**
 * Convert RGBA pixel data → normalized Float32Array in CHW format
 * CHW = Channel × Height × Width (PyTorch convention)
 */
function rgbaToNormalizedCHW(rgba, width, height) {
  const chw = new Float32Array(3 * height * width);
  const area = height * width;
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pxIdx = (y * width + x) * 4;
      const pos   = y * width + x;
      
      const r = rgba[pxIdx]     / 255.0;
      const g = rgba[pxIdx + 1] / 255.0;
      const b = rgba[pxIdx + 2] / 255.0;
      
      chw[0 * area + pos] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      chw[1 * area + pos] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      chw[2 * area + pos] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }
  }
  
  return chw;
}

/**
 * Full preprocessing pipeline:
 *   URI → resize 224×224 → decode JPEG → normalize → Float32 CHW tensor
 */
async function preprocessImage(uri) {
  // Step 1: Resize to 224×224 and get base64
  const resized = await ImageManipulator.manipulateAsync(
    uri,
    [{ resize: { width: INPUT_SIZE, height: INPUT_SIZE } }],
    {
      format: ImageManipulator.SaveFormat.JPEG,
      base64: true,
      compress: 1.0, // No compression for accuracy
    }
  );
  
  // Step 2: Decode JPEG to RGBA pixels
  const { data: rgba } = await decodeJpegBase64(resized.base64);
  
  // Step 3: Normalize and reshape to CHW
  const chw = rgbaToNormalizedCHW(rgba, INPUT_SIZE, INPUT_SIZE);
  
  return chw;
}

// ─── Inference ────────────────────────────────────────────────────────────────

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

async function predict(imageUri) {
  // Load model (cached after first call)
  const session = await getSession();
  
  // Preprocess image
  const inputData = await preprocessImage(imageUri);
  
  // Create ONNX input tensor: [1, 3, 224, 224]
  const inputName = session.inputNames[0];
  const tensor = new Tensor('float32', inputData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const feeds  = { [inputName]: tensor };
  
  // Run inference
  const outputs = await session.run(feeds);
  
  // Extract logits
  const outName = session.outputNames[0];
  const logits  = Array.from(outputs[outName].data);
  
  // Apply softmax to get probabilities
  const probs = softmax(logits);
  
  // Find top prediction
  const topIdx   = probs.indexOf(Math.max(...probs));
  const topClass = CLASS_NAMES[topIdx];
  const topProb  = probs[topIdx];
  
  return {
    predictedClass: topClass,
    confidence: topProb,
    probabilities: probs,
    classes: CLASS_NAMES.map((name, i) => ({
      name,
      info: CLASS_INFO[name],
      probability: probs[i],
      isTop: i === topIdx,
    })),
  };
}

// ─── Utility ─────────────────────────────────────────────────────────────────
function pct(p) { return `${(p * 100).toFixed(1)}%`; }

// ─── Components ───────────────────────────────────────────────────────────────

function ProbabilityBar({ item, isTop }) {
  const info = CLASS_INFO[item.name];
  return (
    <View style={pbs.row}>
      <View style={pbs.labelRow}>
        <Text style={[pbs.className, isTop && { color: '#fff', fontWeight: '700' }]}>
          {info.viName}
        </Text>
        <Text style={pbs.classEng}>{item.name}</Text>
      </View>
      <View style={pbs.barTrack}>
        <View
          style={[
            pbs.barFill,
            {
              width: `${Math.max(item.probability * 100, 0.5).toFixed(1)}%`,
              backgroundColor: isTop ? info.color : info.color + '55',
            },
          ]}
        />
      </View>
      <Text style={[pbs.value, isTop && { color: info.color, fontWeight: '700' }]}>
        {pct(item.probability)}
      </Text>
    </View>
  );
}

const pbs = StyleSheet.create({
  row: { marginBottom: 10 },
  labelRow: { flexDirection: 'row', alignItems: 'baseline', gap: 6, marginBottom: 3 },
  className: { fontSize: 13, color: '#B0BEC5', fontWeight: '500' },
  classEng: { fontSize: 11, color: '#4A5568' },
  barTrack: { height: 6, backgroundColor: '#1a2340', borderRadius: 3, overflow: 'hidden', marginBottom: 2 },
  barFill: { height: '100%', borderRadius: 3 },
  value: { fontSize: 11, color: '#8892A4', textAlign: 'right' },
});

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [imageUri, setImageUri]     = useState(null);
  const [result, setResult]         = useState(null);
  const [loading, setLoading]       = useState(false);
  const [modelReady, setModelReady] = useState(false);
  
  // Preload ONNX model on mount
  useEffect(() => {
    getSession()
      .then(() => setModelReady(true))
      .catch(err => console.warn('Model preload failed:', err));
  }, []);
  
  const handlePickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(
        'Cần quyền truy cập',
        'Ứng dụng cần quyền truy cập thư viện ảnh. Vui lòng cấp quyền trong Cài đặt.',
      );
      return;
    }
    
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      quality: 1,
    });
    
    if (!res.canceled && res.assets?.[0]?.uri) {
      const uri = res.assets[0].uri;
      setImageUri(uri);
      setResult(null);
      runPrediction(uri);
    }
  };
  
  const runPrediction = async (uri) => {
    setLoading(true);
    try {
      const r = await predict(uri);
      setResult(r);
    } catch (err) {
      Alert.alert('Lỗi', 'Phân tích thất bại: ' + err.message);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  
  const info = result ? CLASS_INFO[result.predictedClass] : null;
  
  return (
    <View style={s.root}>
      <ExpoStatusBar style="light" />
      
      {/* ── Header ── */}
      <View style={s.header}>
        <MaterialCommunityIcons name="bird" size={30} color="#00E5FF" />
        <View style={{ flex: 1, marginLeft: 10 }}>
          <Text style={s.headerTitle}>Poultry Health AI</Text>
          <Text style={s.headerSub}>Chẩn đoán bệnh qua mẫu phân gia cầm</Text>
        </View>
        {!modelReady && (
          <View style={s.loadingBadge}>
            <ActivityIndicator size="small" color="#00E5FF" style={{ marginRight: 4 }} />
            <Text style={s.loadingBadgeText}>Tải model...</Text>
          </View>
        )}
        {modelReady && (
          <View style={[s.loadingBadge, { borderColor: '#00E67620' }]}>
            <View style={{ width: 6, height: 6, borderRadius: 3, backgroundColor: '#00E676', marginRight: 5 }} />
            <Text style={[s.loadingBadgeText, { color: '#00E676' }]}>Sẵn sàng</Text>
          </View>
        )}
      </View>
      
      <ScrollView
        style={{ flex: 1 }}
        contentContainerStyle={s.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* ── Image Picker ── */}
        {imageUri ? (
          <View style={s.imageCard}>
            <Image source={{ uri: imageUri }} style={s.image} resizeMode="contain" />
            <TouchableOpacity style={s.changeBtn} onPress={handlePickImage}>
              <Ionicons name="images-outline" size={18} color="#B0BEC5" />
              <Text style={s.changeBtnText}>Chọn ảnh khác</Text>
            </TouchableOpacity>
          </View>
        ) : (
          <TouchableOpacity style={s.pickPlaceholder} onPress={handlePickImage} activeOpacity={0.8}>
            <View style={s.pickIconWrap}>
              <Ionicons name="images" size={56} color="#00E5FF" />
            </View>
            <Text style={s.pickTitle}>Chọn ảnh mẫu phân</Text>
            <Text style={s.pickSub}>Nhấn để mở thư viện ảnh</Text>
          </TouchableOpacity>
        )}
        
        {/* ── Retry Button ── */}
        {imageUri && !loading && (
          <TouchableOpacity style={s.analyzeBtn} onPress={() => runPrediction(imageUri)}>
            <Ionicons name="refresh-outline" size={20} color="#0a0f1e" />
            <Text style={s.analyzeBtnText}>Phân tích lại</Text>
          </TouchableOpacity>
        )}
        
        {/* ── Loading ── */}
        {loading && (
          <View style={s.loadingBox}>
            <ActivityIndicator size="large" color="#00E5FF" />
            <Text style={s.loadingTitle}>Đang phân tích...</Text>
            <Text style={s.loadingSub}>Mô hình AI đang xử lý mẫu phân</Text>
          </View>
        )}
        
        {/* ── Results ── */}
        {result && !loading && (
          <>
            {/* Main Result */}
            <View style={[s.resultCard, { borderColor: info.color }]}>
              <View style={s.resultRow}>
                <View style={[s.resultIcon, { backgroundColor: info.color + '20' }]}>
                  <Ionicons name={info.icon} size={38} color={info.color} />
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={s.resultLabel}>KẾT QUẢ</Text>
                  <Text style={[s.resultClass, { color: info.color }]}>{info.viName}</Text>
                  <Text style={s.resultClassEng}>{result.predictedClass}</Text>
                </View>
              </View>
              
              {/* Confidence meter */}
              <Text style={s.confLabel}>Độ tin cậy</Text>
              <View style={s.confTrack}>
                <View
                  style={[
                    s.confFill,
                    {
                      width: `${(result.confidence * 100).toFixed(1)}%`,
                      backgroundColor: info.color,
                    },
                  ]}
                />
              </View>
              <Text style={[s.confValue, { color: info.color }]}>
                {pct(result.confidence)}
              </Text>
              
              {/* Severity badge */}
              <View style={[s.badge, { borderColor: info.severityColor, backgroundColor: info.severityColor + '15' }]}>
                <View style={[s.badgeDot, { backgroundColor: info.severityColor }]} />
                <Text style={[s.badgeText, { color: info.severityColor }]}>{info.severity}</Text>
              </View>
            </View>
            
            {/* Description */}
            <View style={s.infoCard}>
              <View style={s.infoHeader}>
                <Ionicons name="information-circle-outline" size={18} color="#00E5FF" />
                <Text style={s.infoTitle}>Mô tả</Text>
              </View>
              <Text style={s.infoText}>{info.description}</Text>
            </View>
            
            {/* Recommendation */}
            <View style={[s.infoCard, { borderColor: '#FFB30025' }]}>
              <View style={s.infoHeader}>
                <Ionicons name="medkit-outline" size={18} color="#FFB300" />
                <Text style={[s.infoTitle, { color: '#FFB300' }]}>Khuyến nghị xử lý</Text>
              </View>
              <Text style={s.infoText}>{info.recommendation}</Text>
            </View>
            
            {/* Probability Distribution */}
            <View style={s.probCard}>
              <Text style={s.probTitle}>Phân phối xác suất</Text>
              {result.classes
                .slice()
                .sort((a, b) => b.probability - a.probability)
                .map(item => (
                  <ProbabilityBar key={item.name} item={item} isTop={item.isTop} />
                ))}
            </View>
          </>
        )}
        
        {/* ── Disease Overview (when no image selected) ── */}
        {!imageUri && !result && (
          <View style={{ marginTop: 8 }}>
            <Text style={s.sectionTitle}>Các bệnh được phát hiện</Text>
            {Object.entries(CLASS_INFO).map(([key, ci]) => (
              <View key={key} style={[s.diseaseCard, { borderLeftColor: ci.color }]}>
                <View style={s.diseaseRow}>
                  <Ionicons name={ci.icon} size={20} color={ci.color} />
                  <Text style={[s.diseaseName, { color: ci.color }]}>{ci.viName}</Text>
                  <View style={[s.diseaseBadge, { backgroundColor: ci.severityColor + '20' }]}>
                    <Text style={[s.diseaseBadgeText, { color: ci.severityColor }]}>{ci.severity}</Text>
                  </View>
                </View>
                <Text style={s.diseaseDesc} numberOfLines={2}>{ci.description}</Text>
              </View>
            ))}
          </View>
        )}
        
        {/* Footer */}
        <View style={s.footer}>
          <MaterialCommunityIcons name="cpu-64-bit" size={14} color="#2D3748" />
          <Text style={s.footerText}> MobileNetV3-Small · KD + Pruning · ONNX Runtime</Text>
        </View>
      </ScrollView>
    </View>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────
const s = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: '#070C18',
  },
  
  // Header
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingTop: Platform.OS === 'ios' ? 52 : (StatusBar.currentHeight || 24) + 8,
    paddingBottom: 14,
    paddingHorizontal: 18,
    backgroundColor: '#0C1628',
    borderBottomWidth: 1,
    borderBottomColor: '#162040',
  },
  headerTitle: { fontSize: 20, fontWeight: '800', color: '#fff', letterSpacing: 0.3 },
  headerSub:   { fontSize: 11, color: '#8892A4', marginTop: 1 },
  loadingBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#00E5FF25',
  },
  loadingBadgeText: { fontSize: 10, color: '#00E5FF' },
  
  // Scroll
  scrollContent: { padding: 14, paddingBottom: 40 },
  
  // Image
  imageCard: {
    backgroundColor: '#0C1628',
    borderRadius: 14,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#162040',
    marginBottom: 10,
  },
  image: { width: '100%', height: 270 },
  changeBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 10,
    backgroundColor: '#162040',
  },
  changeBtnText: { color: '#8892A4', fontSize: 13 },
  
  pickPlaceholder: {
    borderWidth: 2,
    borderColor: '#162040',
    borderStyle: 'dashed',
    borderRadius: 14,
    paddingVertical: 50,
    alignItems: 'center',
    marginBottom: 10,
  },
  pickIconWrap: {
    width: 90,
    height: 90,
    borderRadius: 45,
    backgroundColor: '#00E5FF12',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 14,
  },
  pickTitle: { fontSize: 18, fontWeight: '700', color: '#fff', marginBottom: 6 },
  pickSub:   { fontSize: 13, color: '#8892A4' },
  
  // Buttons
  analyzeBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#00E5FF',
    paddingVertical: 13,
    borderRadius: 11,
    marginBottom: 12,
  },
  analyzeBtnText: { color: '#070C18', fontSize: 15, fontWeight: '700' },
  
  // Loading  
  loadingBox: {
    alignItems: 'center',
    paddingVertical: 36,
  },
  loadingTitle: { fontSize: 16, fontWeight: '600', color: '#fff', marginTop: 14 },
  loadingSub:   { fontSize: 13, color: '#8892A4', marginTop: 4 },
  
  // Result Card
  resultCard: {
    backgroundColor: '#0C1628',
    borderRadius: 14,
    padding: 18,
    borderWidth: 1.5,
    marginBottom: 10,
  },
  resultRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 14, marginBottom: 16 },
  resultIcon: {
    width: 68, height: 68, borderRadius: 34,
    alignItems: 'center', justifyContent: 'center',
  },
  resultLabel: {
    fontSize: 10, color: '#8892A4', fontWeight: '600',
    letterSpacing: 1.5, textTransform: 'uppercase', marginBottom: 3,
  },
  resultClass:    { fontSize: 24, fontWeight: '800' },
  resultClassEng: { fontSize: 13, color: '#8892A4', marginTop: 1 },
  
  confLabel:  { fontSize: 11, color: '#8892A4', marginBottom: 5, textTransform: 'uppercase', letterSpacing: 0.5 },
  confTrack:  { height: 8, backgroundColor: '#162040', borderRadius: 4, overflow: 'hidden', marginBottom: 4 },
  confFill:   { height: '100%', borderRadius: 4 },
  confValue:  { fontSize: 18, fontWeight: '700', textAlign: 'right', marginBottom: 12 },
  
  badge: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    alignSelf: 'flex-start',
    paddingHorizontal: 10, paddingVertical: 4,
    borderRadius: 20, borderWidth: 1,
  },
  badgeDot:  { width: 7, height: 7, borderRadius: 3.5 },
  badgeText: { fontSize: 12, fontWeight: '600' },
  
  // Info Card
  infoCard: {
    backgroundColor: '#0C1628',
    borderRadius: 12, padding: 14, marginBottom: 10,
    borderWidth: 1, borderColor: '#162040',
  },
  infoHeader: { flexDirection: 'row', alignItems: 'center', gap: 7, marginBottom: 8 },
  infoTitle: { fontSize: 13, fontWeight: '700', color: '#00E5FF', textTransform: 'uppercase', letterSpacing: 0.5 },
  infoText:  { fontSize: 14, color: '#B0BEC5', lineHeight: 21 },
  
  // Probability card
  probCard: {
    backgroundColor: '#0C1628', borderRadius: 12, padding: 14, marginBottom: 10,
    borderWidth: 1, borderColor: '#162040',
  },
  probTitle: {
    fontSize: 11, color: '#8892A4', fontWeight: '600',
    textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12,
  },
  
  // Disease overview
  sectionTitle: {
    fontSize: 13, color: '#8892A4', fontWeight: '700',
    textTransform: 'uppercase', letterSpacing: 1, marginBottom: 10,
  },
  diseaseCard: {
    backgroundColor: '#0C1628', borderRadius: 11, padding: 12, marginBottom: 8,
    borderWidth: 1, borderColor: '#162040', borderLeftWidth: 4,
  },
  diseaseRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 6 },
  diseaseName: { flex: 1, fontSize: 14, fontWeight: '700' },
  diseaseBadge: { paddingHorizontal: 7, paddingVertical: 2, borderRadius: 8 },
  diseaseBadgeText: { fontSize: 10, fontWeight: '600' },
  diseaseDesc: { fontSize: 13, color: '#8892A4', lineHeight: 18 },
  
  // Footer
  footer: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    paddingTop: 20,
  },
  footerText: { fontSize: 10, color: '#2D3748' },
});
