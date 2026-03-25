import type { InferenceSession } from 'onnxruntime-common';
/**
 * model loading information
 */
interface ModelLoadInfo {
    /**
     * Key for an instance of InferenceSession, which is passed to run() function as parameter.
     */
    readonly key: string;
    /**
     * Get input names of the loaded model.
     */
    readonly inputNames: string[];
    /**
     * Get output names of the loaded model.
     */
    readonly outputNames: string[];
}
/**
 * JSIBlob is a blob object that exchange ArrayBuffer by OnnxruntimeJSIHelper.
 */
export type JSIBlob = {
    blobId: string;
    offset: number;
    size: number;
};
/**
 * Tensor type for react native, which doesn't allow ArrayBuffer in native bridge, so data will be stored as JSIBlob.
 */
interface EncodedTensor {
    /**
     * the dimensions of the tensor.
     */
    readonly dims: readonly number[];
    /**
     * the data type of the tensor.
     */
    readonly type: string;
    /**
     * the JSIBlob object of the buffer data of the tensor.
     * if data is string array, it won't be stored as JSIBlob.
     */
    readonly data: JSIBlob | string[];
}
/**
 * Binding exports a simple synchronized inference session object wrap.
 */
export declare namespace Binding {
    type ModelLoadInfoType = ModelLoadInfo;
    type EncodedTensorType = EncodedTensor;
    type SessionOptions = InferenceSession.SessionOptions;
    type RunOptions = InferenceSession.RunOptions;
    type FeedsType = {
        [name: string]: EncodedTensor;
    };
    type FetchesType = string[];
    type ReturnType = {
        [name: string]: EncodedTensor;
    };
    interface InferenceSession {
        loadModel(modelPath: string, options: SessionOptions): Promise<ModelLoadInfoType>;
        loadModelFromBlob?(blob: JSIBlob, options: SessionOptions): Promise<ModelLoadInfoType>;
        dispose(key: string): Promise<void>;
        run(key: string, feeds: FeedsType, fetches: FetchesType, options: RunOptions): Promise<ReturnType>;
    }
}
export declare const binding: Binding.InferenceSession;
declare global {
    var jsiOnnxruntimeStoreArrayBuffer: ((buffer: ArrayBuffer) => JSIBlob) | undefined;
    var jsiOnnxruntimeResolveArrayBuffer: ((blob: JSIBlob) => ArrayBuffer) | undefined;
}
export declare const jsiHelper: {
    storeArrayBuffer: (buffer: ArrayBuffer) => JSIBlob;
    resolveArrayBuffer: (blob: JSIBlob) => ArrayBuffer;
};
export {};
