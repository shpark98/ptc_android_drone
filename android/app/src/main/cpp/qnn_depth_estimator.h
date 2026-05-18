#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include "qnn_include/QnnCommon.h"
#include "qnn_include/QnnInterface.h"
#include "qnn_include/QnnLog.h"
#include "qnn_include/QnnWrapperUtils.hpp"
#include "qnn_include/HTP/QnnHtpDevice.h"
#include "qnn_include/HTP/QnnHtpGraph.h"
#include "qnn_include/HTP/QnnHtpPerfInfrastructure.h"
#include "qnn_include/QnnGraph.h"
#include "qnn_include/QnnTypes.h"
#include "qnn_include/System/QnnSystemContext.h"
#include "qnn_include/System/QnnSystemInterface.h"

class QnnDepthEstimator {
public:
    QnnDepthEstimator() = default;
    ~QnnDepthEstimator();

    bool initialize(const std::string& nativeLibDir, const std::string& dlcPath);
    bool infer(const float* inputData, float* outputData);
    void destroy();
    void pausePerf();
    void resumePerf();

    uint32_t getInputElementCount() const { return m_inputElementCount; }
    uint32_t getOutputElementCount() const { return m_outputElementCount; }
    bool isInitialized() const { return m_initialized; }

private:
    static void logCallback(const char* fmt, QnnLog_Level_t level,
                            uint64_t timestamp, va_list args);

    // Library handles
    void* m_backendLib = nullptr;
    void* m_modelLib = nullptr;       // libQnnModelDlc.so (DLC fallback path only)
    void* m_systemLib = nullptr;      // libQnnSystem.so (context binary path)

    // QNN handles
    Qnn_LogHandle_t m_logHandle = nullptr;
    Qnn_BackendHandle_t m_backendHandle = nullptr;
    Qnn_DeviceHandle_t m_deviceHandle = nullptr;
    Qnn_ContextHandle_t m_contextHandle = nullptr;

    // QNN interface
    QNN_INTERFACE_VER_TYPE m_qnnFn = QNN_INTERFACE_VER_TYPE_INIT;

    // System interface (for context binary metadata)
    QNN_SYSTEM_INTERFACE_VER_TYPE m_sysFn = QNN_SYSTEM_INTERFACE_VER_TYPE_INIT;
    QnnSystemContext_Handle_t m_sysCtxHandle = nullptr;

    // Graph info from composeGraphsFromDlc (DLC path)
    qnn_wrapper_api::GraphInfoPtr_t* m_graphsInfo = nullptr;
    uint32_t m_numGraphs = 0;

    // Graph state for context binary path
    bool m_isContextBinary = false;
    Qnn_GraphHandle_t m_graphHandle = nullptr;        // single graph (retrieved from .bin)
    std::vector<Qnn_Tensor_t> m_binInputTensors;       // owned copies from system context
    std::vector<Qnn_Tensor_t> m_binOutputTensors;

    // Execution tensors with allocated buffers
    std::vector<Qnn_Tensor_t> m_execInputTensors;
    std::vector<Qnn_Tensor_t> m_execOutputTensors;
    std::vector<std::vector<uint8_t>> m_inputBuffers;
    std::vector<std::vector<uint8_t>> m_outputBuffers;

    uint32_t m_inputElementCount = 0;
    uint32_t m_outputElementCount = 0;
    bool m_initialized = false;

    // HTP Performance Infrastructure
    QnnHtpDevice_PerfInfrastructure_t m_perfInfra = QNN_HTP_DEVICE_PERF_INFRASTRUCTURE_INIT;
    uint32_t m_powerConfigId = 0;
    bool m_perfConfigured = false;

    // Quantization info for I/O tensors
    struct QuantInfo {
        bool isQuantized = false;
        Qnn_DataType_t dataType = QNN_DATATYPE_FLOAT_32;
        float scale = 1.0f;
        int32_t offset = 0;
    };
    QuantInfo m_inputQuant;
    QuantInfo m_outputQuant;

    // Helpers
    static uint32_t getDataTypeSize(Qnn_DataType_t dataType);
    uint32_t calculateTensorSize(const Qnn_Tensor_t& tensor);
    bool setupExecutionTensors();
    bool setupExecutionTensorsFromBinaryInfo(const Qnn_Tensor_t* inputs, uint32_t numInputs,
                                              const Qnn_Tensor_t* outputs, uint32_t numOutputs);
    bool setupPerfConfig();
    QuantInfo extractQuantInfo(const Qnn_Tensor_t& tensor);

    // Context binary loading path (fast — pre-compiled for target SoC).
    // Returns false to indicate caller should fall back to DLC compose path.
    bool initFromContextBinary(const std::string& nativeLibDir, const std::string& binPath);
    bool initFromDlc(const std::string& nativeLibDir, const std::string& dlcPath);
};
