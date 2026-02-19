#include "qnn_depth_estimator.h"

#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <string>
#include <android/log.h>

#include "qnn_include/QnnTypeMacros.hpp"

#define LOG_TAG "QnnDepthEstimator"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

// Function pointer types for dlsym
typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(
    const QnnInterface_t*** providerList, uint32_t* numProviders);

typedef qnn_wrapper_api::ModelError_t (*ComposeGraphsFromDlcFn_t)(
    Qnn_BackendHandle_t backendHandle,
    QNN_INTERFACE_VER_TYPE qnnInterface,
    Qnn_ContextHandle_t contextHandle,
    const qnn_wrapper_api::GraphConfigInfo_t** graphsConfigInfo,
    const char* dlcPath,
    const uint32_t numGraphsConfigInfo,
    qnn_wrapper_api::GraphInfoPtr_t** graphsInfo,
    uint32_t* numGraphsInfo,
    bool debug,
    QnnLog_Callback_t logCallback,
    QnnLog_Level_t maxLogLevel);

typedef qnn_wrapper_api::ModelError_t (*FreeGraphsInfoFn_t)(
    qnn_wrapper_api::GraphInfoPtr_t** graphsInfo, uint32_t numGraphs);

// --- Implementation ---

QnnDepthEstimator::~QnnDepthEstimator() {
    destroy();
}

void QnnDepthEstimator::logCallback(const char* fmt, QnnLog_Level_t level,
                                     uint64_t /*timestamp*/, va_list args) {
    int androidLevel = ANDROID_LOG_VERBOSE;
    switch (level) {
        case QNN_LOG_LEVEL_ERROR:   androidLevel = ANDROID_LOG_ERROR; break;
        case QNN_LOG_LEVEL_WARN:    androidLevel = ANDROID_LOG_WARN;  break;
        case QNN_LOG_LEVEL_INFO:    androidLevel = ANDROID_LOG_INFO;  break;
        case QNN_LOG_LEVEL_VERBOSE: androidLevel = ANDROID_LOG_VERBOSE; break;
        case QNN_LOG_LEVEL_DEBUG:   androidLevel = ANDROID_LOG_DEBUG; break;
        default: break;
    }
    __android_log_vprint(androidLevel, "QNN", fmt, args);
}

uint32_t QnnDepthEstimator::getDataTypeSize(Qnn_DataType_t dataType) {
    switch (dataType) {
        case QNN_DATATYPE_FLOAT_64: return 8;
        case QNN_DATATYPE_FLOAT_32:
        case QNN_DATATYPE_INT_32:
        case QNN_DATATYPE_UINT_32:
        case QNN_DATATYPE_SFIXED_POINT_32:
        case QNN_DATATYPE_UFIXED_POINT_32: return 4;
        case QNN_DATATYPE_FLOAT_16:
        case QNN_DATATYPE_INT_16:
        case QNN_DATATYPE_UINT_16:
        case QNN_DATATYPE_SFIXED_POINT_16:
        case QNN_DATATYPE_UFIXED_POINT_16: return 2;
        case QNN_DATATYPE_INT_8:
        case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_SFIXED_POINT_8:
        case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_BOOL_8: return 1;
        default: return 0;
    }
}

uint32_t QnnDepthEstimator::calculateTensorSize(const Qnn_Tensor_t& tensor) {
    uint32_t rank = QNN_TENSOR_GET_RANK(tensor);
    uint32_t* dims = QNN_TENSOR_GET_DIMENSIONS(tensor);
    if (rank == 0 || dims == nullptr) return 0;

    uint32_t numElements = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        numElements *= dims[i];
    }
    return numElements * getDataTypeSize(QNN_TENSOR_GET_DATA_TYPE(tensor));
}

QnnDepthEstimator::QuantInfo QnnDepthEstimator::extractQuantInfo(const Qnn_Tensor_t& tensor) {
    QuantInfo qi;
    qi.dataType = QNN_TENSOR_GET_DATA_TYPE(tensor);

    // Check if data type is quantized (fixed point)
    switch (qi.dataType) {
        case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_UFIXED_POINT_16:
        case QNN_DATATYPE_UFIXED_POINT_32:
        case QNN_DATATYPE_SFIXED_POINT_8:
        case QNN_DATATYPE_SFIXED_POINT_16:
        case QNN_DATATYPE_SFIXED_POINT_32:
            qi.isQuantized = true;
            break;
        default:
            qi.isQuantized = false;
            return qi;
    }

    Qnn_QuantizeParams_t qParams = QNN_TENSOR_GET_QUANT_PARAMS(tensor);
    if (qParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        qi.scale = qParams.scaleOffsetEncoding.scale;
        qi.offset = qParams.scaleOffsetEncoding.offset;
        LOGI("  Quant params: scale=%e, offset=%d", qi.scale, qi.offset);
    } else if (qParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET) {
        qi.scale = qParams.bwScaleOffsetEncoding.scale;
        qi.offset = qParams.bwScaleOffsetEncoding.offset;
        LOGI("  Quant params (BW): scale=%e, offset=%d, bw=%u",
             qi.scale, qi.offset, qParams.bwScaleOffsetEncoding.bitwidth);
    } else {
        LOGE("  Unsupported quantization encoding: %d", qParams.quantizationEncoding);
        qi.isQuantized = false;
    }
    return qi;
}

bool QnnDepthEstimator::setupPerfConfig() {
    // Get device infrastructure (contains perf API function pointers)
    if (!m_qnnFn.deviceGetInfrastructure) {
        LOGI("deviceGetInfrastructure not available, skipping perf config");
        return false;
    }

    // QnnDevice_Infrastructure_t is a pointer type (typedef struct _* QnnDevice_Infrastructure_t)
    // deviceGetInfrastructure takes a pointer-to-pointer and fills it
    QnnDevice_Infrastructure_t devInfra = nullptr;
    auto err = m_qnnFn.deviceGetInfrastructure(&devInfra);
    if (err != QNN_SUCCESS) {
        LOGE("deviceGetInfrastructure failed: %lu", (unsigned long)err);
        return false;
    }
    if (!devInfra) {
        LOGE("deviceGetInfrastructure returned null");
        return false;
    }

    auto* htpInfra = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(devInfra);
    m_perfInfra = htpInfra->perfInfra;
    if (!m_perfInfra.createPowerConfigId || !m_perfInfra.setPowerConfig) {
        LOGE("Perf infrastructure function pointers are null");
        return false;
    }
    LOGI("Got HTP performance infrastructure");

    // Create power config ID (deviceId=0, coreId=0 for default)
    err = m_perfInfra.createPowerConfigId(0, 0, &m_powerConfigId);
    if (err != QNN_SUCCESS) {
        LOGE("createPowerConfigId failed: %lu", (unsigned long)err);
        return false;
    }
    LOGI("Created power config ID: %u", m_powerConfigId);

    // Configure DCVS v3 with TURBO voltage corners and PERFORMANCE_MODE
    QnnHtpPerfInfrastructure_PowerConfig_t dcvsConfig;
    memset(&dcvsConfig, 0, sizeof(dcvsConfig));
    dcvsConfig.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    dcvsConfig.dcvsV3Config.contextId = m_powerConfigId;
    dcvsConfig.dcvsV3Config.setDcvsEnable = 1;
    dcvsConfig.dcvsV3Config.dcvsEnable = 0;  // Disable DCVS (lock to max clocks)
    dcvsConfig.dcvsV3Config.powerMode =
        QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    dcvsConfig.dcvsV3Config.setSleepLatency = 1;
    dcvsConfig.dcvsV3Config.sleepLatency = 40;  // 40us sleep latency
    dcvsConfig.dcvsV3Config.setSleepDisable = 1;
    dcvsConfig.dcvsV3Config.sleepDisable = 1;  // Disable sleep
    dcvsConfig.dcvsV3Config.setBusParams = 1;
    dcvsConfig.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO;
    dcvsConfig.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
    dcvsConfig.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
    dcvsConfig.dcvsV3Config.setCoreParams = 1;
    dcvsConfig.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO;
    dcvsConfig.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
    dcvsConfig.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;

    // Configure RPC polling for lower latency (V69+)
    QnnHtpPerfInfrastructure_PowerConfig_t rpcPollingConfig;
    memset(&rpcPollingConfig, 0, sizeof(rpcPollingConfig));
    rpcPollingConfig.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    rpcPollingConfig.rpcPollingTimeConfig = 9999;  // Max polling time (us)

    // Configure RPC control latency
    QnnHtpPerfInfrastructure_PowerConfig_t rpcLatencyConfig;
    memset(&rpcLatencyConfig, 0, sizeof(rpcLatencyConfig));
    rpcLatencyConfig.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpcLatencyConfig.rpcControlLatencyConfig = 100;  // 100us RPC latency

    // Apply all power configs (NULL terminated array of pointers)
    const QnnHtpPerfInfrastructure_PowerConfig_t* configArr[] = {
        &dcvsConfig,
        &rpcPollingConfig,
        &rpcLatencyConfig,
        nullptr
    };

    err = m_perfInfra.setPowerConfig(m_powerConfigId, configArr);
    if (err != QNN_SUCCESS) {
        LOGE("setPowerConfig failed: %lu", (unsigned long)err);
        return false;
    }

    m_perfConfigured = true;
    LOGI("HTP performance configured: TURBO clocks, PERFORMANCE_MODE, RPC polling enabled");
    return true;
}

bool QnnDepthEstimator::initialize(const std::string& nativeLibDir,
                                    const std::string& dlcPath) {
    if (m_initialized) {
        LOGI("Already initialized");
        return true;
    }

    LOGI("Initializing QNN Depth Estimator (HTP backend)...");
    LOGI("  Native lib dir: %s", nativeLibDir.c_str());
    LOGI("  DLC path: %s", dlcPath.c_str());

    // Set ADSP_LIBRARY_PATH so HTP can find skel libraries
    std::string adspPath = nativeLibDir + ";/dsp";
    setenv("ADSP_LIBRARY_PATH", adspPath.c_str(), 1);
    LOGI("ADSP_LIBRARY_PATH=%s", adspPath.c_str());

    // 1. Load HTP backend library
    {
        std::string backendPath = nativeLibDir + "/libQnnHtp.so";
        m_backendLib = dlopen(backendPath.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!m_backendLib) {
            LOGE("Failed to load libQnnHtp.so: %s", dlerror());
            return false;
        }
        LOGI("Loaded libQnnHtp.so");
    }

    // 2. Get QNN interface from backend
    auto getProviders = reinterpret_cast<QnnInterfaceGetProvidersFn_t>(
        dlsym(m_backendLib, "QnnInterface_getProviders"));
    if (!getProviders) {
        LOGE("QnnInterface_getProviders not found: %s", dlerror());
        return false;
    }

    const QnnInterface_t** providerList = nullptr;
    uint32_t numProviders = 0;
    if (getProviders(&providerList, &numProviders) != QNN_SUCCESS ||
        numProviders == 0 || providerList == nullptr) {
        LOGE("QnnInterface_getProviders failed");
        return false;
    }
    LOGI("Got %u QNN interface provider(s)", numProviders);
    m_qnnFn = providerList[0]->QNN_INTERFACE_VER_NAME;

    // 3. Create logger
    if (m_qnnFn.logCreate) {
        auto err = m_qnnFn.logCreate(logCallback, QNN_LOG_LEVEL_INFO, &m_logHandle);
        if (err != QNN_SUCCESS) {
            LOGE("logCreate failed (non-fatal): %lu", (unsigned long)err);
            m_logHandle = nullptr;
        } else {
            LOGI("QNN logging initialized");
        }
    }

    // 4. Create HTP backend
    if (!m_qnnFn.backendCreate) {
        LOGE("backendCreate is null");
        return false;
    }

    auto err = m_qnnFn.backendCreate(m_logHandle, nullptr, &m_backendHandle);
    if (err != QNN_SUCCESS) {
        LOGE("backendCreate failed: %lu", (unsigned long)err);
        return false;
    }
    LOGI("QNN HTP backend created");

    // 5. Create device
    if (m_qnnFn.deviceCreate) {
        err = m_qnnFn.deviceCreate(m_logHandle, nullptr, &m_deviceHandle);
        if (err != QNN_SUCCESS) {
            LOGE("deviceCreate failed (non-fatal): %lu", (unsigned long)err);
            m_deviceHandle = nullptr;
        } else {
            LOGI("QNN device created");
        }
    }

    // 5.5. Setup HTP performance mode (TURBO clocks + RPC polling)
    if (!setupPerfConfig()) {
        LOGI("HTP perf config failed (non-fatal), continuing with defaults");
    }

    // 6. Create context
    if (!m_qnnFn.contextCreate) {
        LOGE("contextCreate is null");
        return false;
    }
    err = m_qnnFn.contextCreate(m_backendHandle, m_deviceHandle, nullptr, &m_contextHandle);
    if (err != QNN_SUCCESS) {
        LOGE("contextCreate failed: %lu", (unsigned long)err);
        return false;
    }
    LOGI("QNN context created");

    // 7. Load DLC model library
    std::string modelLibPath = nativeLibDir + "/libQnnModelDlc.so";
    m_modelLib = dlopen(modelLibPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!m_modelLib) {
        LOGE("Failed to load libQnnModelDlc.so: %s", dlerror());
        return false;
    }
    LOGI("Loaded libQnnModelDlc.so");

    auto composeGraphs = reinterpret_cast<ComposeGraphsFromDlcFn_t>(
        dlsym(m_modelLib, "QnnModel_composeGraphsFromDlc"));
    if (!composeGraphs) {
        LOGE("QnnModel_composeGraphsFromDlc not found: %s", dlerror());
        return false;
    }

    // 8. Compose graphs from DLC
    {
        FILE* fp = fopen(dlcPath.c_str(), "rb");
        if (!fp) {
            LOGE("Cannot open DLC file: %s (errno=%d: %s)", dlcPath.c_str(), errno, strerror(errno));
            return false;
        }
        fseek(fp, 0, SEEK_END);
        long fileSize = ftell(fp);
        fclose(fp);
        LOGI("DLC file verified: %s (%ld bytes)", dlcPath.c_str(), fileSize);
    }

    LOGI("Composing graphs from DLC: %s", dlcPath.c_str());
    auto modelErr = composeGraphs(
        m_backendHandle,
        m_qnnFn,
        m_contextHandle,
        nullptr,
        dlcPath.c_str(),
        0,
        &m_graphsInfo,
        &m_numGraphs,
        false,
        logCallback,
        QNN_LOG_LEVEL_INFO
    );
    if (modelErr != qnn_wrapper_api::MODEL_NO_ERROR) {
        LOGE("composeGraphsFromDlc failed: %d", modelErr);
        return false;
    }
    LOGI("Composed %u graph(s) from DLC", m_numGraphs);

    if (m_numGraphs == 0 || m_graphsInfo == nullptr) {
        LOGE("No graphs loaded from DLC");
        return false;
    }

    // 9. Finalize graphs for HTP execution
    for (uint32_t i = 0; i < m_numGraphs; ++i) {
        auto* gi = m_graphsInfo[i];
        LOGI("Graph[%u]: name='%s', inputs=%u, outputs=%u",
             i, gi->graphName, gi->numInputTensors, gi->numOutputTensors);

        if (!m_qnnFn.graphFinalize) {
            LOGE("graphFinalize is null");
            return false;
        }
        err = m_qnnFn.graphFinalize(gi->graph, nullptr, nullptr);
        if (err != QNN_SUCCESS) {
            LOGE("graphFinalize failed for '%s': %lu", gi->graphName, (unsigned long)err);
            return false;
        }
        LOGI("Graph '%s' finalized for HTP", gi->graphName);
    }

    // 10. Setup execution tensor buffers
    if (!setupExecutionTensors()) {
        LOGE("Failed to setup execution tensors");
        return false;
    }

    m_initialized = true;
    LOGI("QNN Depth Estimator initialized successfully (HTP backend)");
    if (m_inputQuant.isQuantized) {
        LOGI("  Input: quantized %s, scale=%e, offset=%d",
             m_inputQuant.dataType == QNN_DATATYPE_UFIXED_POINT_16 ? "uint16" : "other",
             m_inputQuant.scale, m_inputQuant.offset);
    } else {
        LOGI("  Input: float32");
    }
    if (m_outputQuant.isQuantized) {
        LOGI("  Output: quantized %s, scale=%e, offset=%d",
             m_outputQuant.dataType == QNN_DATATYPE_UFIXED_POINT_16 ? "uint16" : "other",
             m_outputQuant.scale, m_outputQuant.offset);
    } else {
        LOGI("  Output: float32");
    }
    return true;
}

bool QnnDepthEstimator::setupExecutionTensors() {
    auto* gi = m_graphsInfo[0];

    // Setup input tensors
    m_execInputTensors.resize(gi->numInputTensors);
    m_inputBuffers.resize(gi->numInputTensors);
    m_inputElementCount = 0;

    for (uint32_t i = 0; i < gi->numInputTensors; ++i) {
        m_execInputTensors[i] = gi->inputTensors[i];

        uint32_t tensorSize = calculateTensorSize(gi->inputTensors[i]);
        if (tensorSize == 0) {
            LOGE("Input tensor %u has zero size", i);
            return false;
        }
        m_inputBuffers[i].resize(tensorSize, 0);

        QNN_TENSOR_SET_MEM_TYPE(m_execInputTensors[i], QNN_TENSORMEMTYPE_RAW);
        Qnn_ClientBuffer_t clientBuf = {m_inputBuffers[i].data(), tensorSize};
        QNN_TENSOR_SET_CLIENT_BUF(m_execInputTensors[i], clientBuf);

        uint32_t rank = QNN_TENSOR_GET_RANK(gi->inputTensors[i]);
        uint32_t* dims = QNN_TENSOR_GET_DIMENSIONS(gi->inputTensors[i]);
        uint32_t elements = 1;
        for (uint32_t d = 0; d < rank; ++d) elements *= dims[d];
        m_inputElementCount += elements;

        std::string dimStr;
        for (uint32_t d = 0; d < rank; ++d) {
            if (d > 0) dimStr += "x";
            dimStr += std::to_string(dims[d]);
        }
        LOGI("Input[%u]: name='%s', dims=[%s], dtype=0x%x, bytes=%u",
             i, QNN_TENSOR_GET_NAME(gi->inputTensors[i]),
             dimStr.c_str(), QNN_TENSOR_GET_DATA_TYPE(gi->inputTensors[i]), tensorSize);

        // Extract quantization info for first input
        if (i == 0) {
            m_inputQuant = extractQuantInfo(gi->inputTensors[i]);
        }
    }

    // Setup output tensors
    m_execOutputTensors.resize(gi->numOutputTensors);
    m_outputBuffers.resize(gi->numOutputTensors);
    m_outputElementCount = 0;

    for (uint32_t i = 0; i < gi->numOutputTensors; ++i) {
        m_execOutputTensors[i] = gi->outputTensors[i];

        uint32_t tensorSize = calculateTensorSize(gi->outputTensors[i]);
        if (tensorSize == 0) {
            LOGE("Output tensor %u has zero size", i);
            return false;
        }
        m_outputBuffers[i].resize(tensorSize, 0);

        QNN_TENSOR_SET_MEM_TYPE(m_execOutputTensors[i], QNN_TENSORMEMTYPE_RAW);
        Qnn_ClientBuffer_t clientBuf = {m_outputBuffers[i].data(), tensorSize};
        QNN_TENSOR_SET_CLIENT_BUF(m_execOutputTensors[i], clientBuf);

        uint32_t rank = QNN_TENSOR_GET_RANK(gi->outputTensors[i]);
        uint32_t* dims = QNN_TENSOR_GET_DIMENSIONS(gi->outputTensors[i]);
        uint32_t elements = 1;
        for (uint32_t d = 0; d < rank; ++d) elements *= dims[d];
        m_outputElementCount += elements;

        std::string dimStr;
        for (uint32_t d = 0; d < rank; ++d) {
            if (d > 0) dimStr += "x";
            dimStr += std::to_string(dims[d]);
        }
        LOGI("Output[%u]: name='%s', dims=[%s], dtype=0x%x, bytes=%u",
             i, QNN_TENSOR_GET_NAME(gi->outputTensors[i]),
             dimStr.c_str(), QNN_TENSOR_GET_DATA_TYPE(gi->outputTensors[i]), tensorSize);

        // Extract quantization info for first output
        if (i == 0) {
            m_outputQuant = extractQuantInfo(gi->outputTensors[i]);
        }
    }

    LOGI("Total input elements: %u, output elements: %u",
         m_inputElementCount, m_outputElementCount);
    return true;
}

bool QnnDepthEstimator::infer(const float* inputData, float* outputData) {
    if (!m_initialized) {
        LOGE("Not initialized");
        return false;
    }

    // Copy input data into tensor buffer, handling quantization if needed
    if (m_inputQuant.isQuantized) {
        // Float -> quantized conversion
        uint32_t dtSize = getDataTypeSize(m_inputQuant.dataType);
        uint32_t expectedBytes = m_inputElementCount * dtSize;
        if (m_inputBuffers.empty() || m_inputBuffers[0].size() < expectedBytes) {
            LOGE("Input buffer size mismatch (quant): need %u, have %zu",
                 expectedBytes, m_inputBuffers.empty() ? 0 : m_inputBuffers[0].size());
            return false;
        }

        float invScale = 1.0f / m_inputQuant.scale;
        int32_t offset = m_inputQuant.offset;

        if (m_inputQuant.dataType == QNN_DATATYPE_UFIXED_POINT_16) {
            auto* dst = reinterpret_cast<uint16_t*>(m_inputBuffers[0].data());
            for (uint32_t i = 0; i < m_inputElementCount; ++i) {
                // QNN convention: real_value = (quantized + offset) * scale
                // So: quantized = real_value / scale - offset
                float qVal = inputData[i] * invScale - static_cast<float>(offset);
                int32_t q = static_cast<int32_t>(roundf(qVal));
                dst[i] = static_cast<uint16_t>(std::max(0, std::min(65535, q)));
            }
        } else if (m_inputQuant.dataType == QNN_DATATYPE_UFIXED_POINT_8) {
            auto* dst = m_inputBuffers[0].data();
            for (uint32_t i = 0; i < m_inputElementCount; ++i) {
                float qVal = inputData[i] * invScale - static_cast<float>(offset);
                int32_t q = static_cast<int32_t>(roundf(qVal));
                dst[i] = static_cast<uint8_t>(std::max(0, std::min(255, q)));
            }
        } else {
            LOGE("Unsupported quantized input data type: 0x%x", m_inputQuant.dataType);
            return false;
        }
    } else {
        // Float32 direct copy
        uint32_t inputBytes = m_inputElementCount * sizeof(float);
        if (m_inputBuffers.empty() || m_inputBuffers[0].size() < inputBytes) {
            LOGE("Input buffer size mismatch: need %u, have %zu",
                 inputBytes, m_inputBuffers.empty() ? 0 : m_inputBuffers[0].size());
            return false;
        }
        memcpy(m_inputBuffers[0].data(), inputData, inputBytes);
    }

    // Execute
    auto* gi = m_graphsInfo[0];
    auto err = m_qnnFn.graphExecute(
        gi->graph,
        m_execInputTensors.data(),
        static_cast<uint32_t>(m_execInputTensors.size()),
        m_execOutputTensors.data(),
        static_cast<uint32_t>(m_execOutputTensors.size()),
        nullptr, nullptr
    );
    if (err != QNN_SUCCESS) {
        LOGE("graphExecute failed: %lu", (unsigned long)err);
        return false;
    }

    // Copy output data, handling dequantization if needed
    if (m_outputQuant.isQuantized) {
        float scale = m_outputQuant.scale;
        int32_t offset = m_outputQuant.offset;

        if (m_outputQuant.dataType == QNN_DATATYPE_UFIXED_POINT_16) {
            auto* src = reinterpret_cast<const uint16_t*>(m_outputBuffers[0].data());
            for (uint32_t i = 0; i < m_outputElementCount; ++i) {
                // QNN convention: real_value = (quantized + offset) * scale
                outputData[i] = (static_cast<float>(src[i]) + static_cast<float>(offset)) * scale;
            }
        } else if (m_outputQuant.dataType == QNN_DATATYPE_UFIXED_POINT_8) {
            auto* src = m_outputBuffers[0].data();
            for (uint32_t i = 0; i < m_outputElementCount; ++i) {
                outputData[i] = (static_cast<float>(src[i]) + static_cast<float>(offset)) * scale;
            }
        } else {
            LOGE("Unsupported quantized output data type: 0x%x", m_outputQuant.dataType);
            return false;
        }
    } else {
        // Float32 direct copy
        uint32_t outputBytes = m_outputElementCount * sizeof(float);
        if (m_outputBuffers.empty() || m_outputBuffers[0].size() < outputBytes) {
            LOGE("Output buffer size mismatch");
            return false;
        }
        memcpy(outputData, m_outputBuffers[0].data(), outputBytes);
    }

    return true;
}

void QnnDepthEstimator::pausePerf() {
    if (!m_perfConfigured) return;

    // Release TURBO clocks — let DSP go to low power
    QnnHtpPerfInfrastructure_PowerConfig_t dcvsConfig;
    memset(&dcvsConfig, 0, sizeof(dcvsConfig));
    dcvsConfig.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    dcvsConfig.dcvsV3Config.contextId = m_powerConfigId;
    dcvsConfig.dcvsV3Config.setDcvsEnable = 1;
    dcvsConfig.dcvsV3Config.dcvsEnable = 1;  // Re-enable DCVS (adaptive clocking)
    dcvsConfig.dcvsV3Config.powerMode =
        QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;
    dcvsConfig.dcvsV3Config.setSleepDisable = 1;
    dcvsConfig.dcvsV3Config.sleepDisable = 0;  // Allow sleep
    dcvsConfig.dcvsV3Config.setBusParams = 1;
    dcvsConfig.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;
    dcvsConfig.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;
    dcvsConfig.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;
    dcvsConfig.dcvsV3Config.setCoreParams = 1;
    dcvsConfig.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;
    dcvsConfig.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;
    dcvsConfig.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;

    // Disable RPC polling
    QnnHtpPerfInfrastructure_PowerConfig_t rpcPollingConfig;
    memset(&rpcPollingConfig, 0, sizeof(rpcPollingConfig));
    rpcPollingConfig.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    rpcPollingConfig.rpcPollingTimeConfig = 0;  // Disable polling

    const QnnHtpPerfInfrastructure_PowerConfig_t* configArr[] = {
        &dcvsConfig, &rpcPollingConfig, nullptr
    };

    if (m_perfInfra.setPowerConfig) {
        m_perfInfra.setPowerConfig(m_powerConfigId, configArr);
    }
    LOGI("HTP perf paused: DSP set to power-saver mode");
}

void QnnDepthEstimator::resumePerf() {
    if (!m_perfConfigured) return;

    // Restore TURBO clocks
    QnnHtpPerfInfrastructure_PowerConfig_t dcvsConfig;
    memset(&dcvsConfig, 0, sizeof(dcvsConfig));
    dcvsConfig.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    dcvsConfig.dcvsV3Config.contextId = m_powerConfigId;
    dcvsConfig.dcvsV3Config.setDcvsEnable = 1;
    dcvsConfig.dcvsV3Config.dcvsEnable = 0;  // Disable DCVS (lock to max)
    dcvsConfig.dcvsV3Config.powerMode =
        QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    dcvsConfig.dcvsV3Config.setSleepLatency = 1;
    dcvsConfig.dcvsV3Config.sleepLatency = 40;
    dcvsConfig.dcvsV3Config.setSleepDisable = 1;
    dcvsConfig.dcvsV3Config.sleepDisable = 1;
    dcvsConfig.dcvsV3Config.setBusParams = 1;
    dcvsConfig.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO;
    dcvsConfig.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
    dcvsConfig.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
    dcvsConfig.dcvsV3Config.setCoreParams = 1;
    dcvsConfig.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO;
    dcvsConfig.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
    dcvsConfig.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;

    QnnHtpPerfInfrastructure_PowerConfig_t rpcPollingConfig;
    memset(&rpcPollingConfig, 0, sizeof(rpcPollingConfig));
    rpcPollingConfig.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    rpcPollingConfig.rpcPollingTimeConfig = 9999;

    QnnHtpPerfInfrastructure_PowerConfig_t rpcLatencyConfig;
    memset(&rpcLatencyConfig, 0, sizeof(rpcLatencyConfig));
    rpcLatencyConfig.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpcLatencyConfig.rpcControlLatencyConfig = 100;

    const QnnHtpPerfInfrastructure_PowerConfig_t* configArr[] = {
        &dcvsConfig, &rpcPollingConfig, &rpcLatencyConfig, nullptr
    };

    if (m_perfInfra.setPowerConfig) {
        m_perfInfra.setPowerConfig(m_powerConfigId, configArr);
    }
    LOGI("HTP perf resumed: TURBO clocks, RPC polling restored");
}

void QnnDepthEstimator::destroy() {
    if (!m_initialized && !m_backendLib) return;
    LOGI("Destroying QNN Depth Estimator...");

    m_execInputTensors.clear();
    m_execOutputTensors.clear();
    m_inputBuffers.clear();
    m_outputBuffers.clear();

    // Release power config
    if (m_perfConfigured && m_perfInfra.destroyPowerConfigId) {
        m_perfInfra.destroyPowerConfigId(m_powerConfigId);
        m_powerConfigId = 0;
        m_perfConfigured = false;
        LOGI("Power config released");
    }

    // Free context (invalidates graph handles)
    if (m_contextHandle && m_qnnFn.contextFree) {
        m_qnnFn.contextFree(m_contextHandle, nullptr);
        m_contextHandle = nullptr;
    }

    // Free graphsInfo allocated by composeGraphsFromDlc
    if (m_graphsInfo && m_modelLib) {
        auto freeGraphs = reinterpret_cast<FreeGraphsInfoFn_t>(
            dlsym(m_modelLib, "_ZN15qnn_wrapper_api14freeGraphsInfoEPPPNS_9GraphInfoEj"));
        if (freeGraphs) {
            freeGraphs(&m_graphsInfo, m_numGraphs);
        } else {
            for (uint32_t i = 0; i < m_numGraphs; ++i) {
                if (m_graphsInfo[i]) {
                    free(m_graphsInfo[i]->graphName);
                    for (uint32_t j = 0; j < m_graphsInfo[i]->numInputTensors; ++j) {
                        free((void*)QNN_TENSOR_GET_NAME(m_graphsInfo[i]->inputTensors[j]));
                        free(QNN_TENSOR_GET_DIMENSIONS(m_graphsInfo[i]->inputTensors[j]));
                    }
                    free(m_graphsInfo[i]->inputTensors);
                    for (uint32_t j = 0; j < m_graphsInfo[i]->numOutputTensors; ++j) {
                        free((void*)QNN_TENSOR_GET_NAME(m_graphsInfo[i]->outputTensors[j]));
                        free(QNN_TENSOR_GET_DIMENSIONS(m_graphsInfo[i]->outputTensors[j]));
                    }
                    free(m_graphsInfo[i]->outputTensors);
                }
            }
            free(*m_graphsInfo);
            free(m_graphsInfo);
        }
        m_graphsInfo = nullptr;
        m_numGraphs = 0;
    }

    if (m_deviceHandle && m_qnnFn.deviceFree) {
        m_qnnFn.deviceFree(m_deviceHandle);
        m_deviceHandle = nullptr;
    }
    if (m_backendHandle && m_qnnFn.backendFree) {
        m_qnnFn.backendFree(m_backendHandle);
        m_backendHandle = nullptr;
    }
    if (m_logHandle && m_qnnFn.logFree) {
        m_qnnFn.logFree(m_logHandle);
        m_logHandle = nullptr;
    }

    if (m_modelLib) { dlclose(m_modelLib); m_modelLib = nullptr; }
    if (m_backendLib) { dlclose(m_backendLib); m_backendLib = nullptr; }

    m_initialized = false;
    LOGI("QNN Depth Estimator destroyed");
}
