import time

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt


def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        binding_shape = engine.get_binding_shape(binding)
        size = trt.volume(binding_shape) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))

        if engine.binding_is_input(binding):
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})
    return inputs, outputs, bindings, stream


def preprocess_image(image_path, input_shape):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (input_shape[2], input_shape[1]))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_transposed = np.transpose(image_rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(image_transposed, axis=0)


def postprocess_output(output, output_shape):
    return output.reshape(output_shape)


def infer(engine_path, image_path):
    engine = load_engine(engine_path)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()

    input_shape = engine.get_binding_shape(0)
    input_data = preprocess_image(image_path, input_shape)

    np.copyto(inputs[0]["host"], input_data.ravel())

    cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)

    start_time = time.time()
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    stream.synchronize()
    end_time = time.time()

    cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
    stream.synchronize()

    output = postprocess_output(outputs[0]["host"], engine.get_binding_shape(1))

    print(f"Inference Time: {end_time - start_time:.4f} seconds")
    return output


if __name__ == "__main__":
    print(trt.__version__)
    engine_file = r"D:\Schule\JG1\WORLD ROBOTICS OLYMPIAD 2025\Code\LanePilot\assets\trained_models\vehicle_detection\vehicle_detection.engine"
    image_file = r"D:\Schule\JG1\WORLD ROBOTICS OLYMPIAD 2025\Code\LanePilot\.cache\vehicle_detection\test\images\image_13_png.rf.9b15756c95201114b503cb5e06ac95b5.jpg"
    output = infer(engine_file, image_file)
    print("Output:", output)
