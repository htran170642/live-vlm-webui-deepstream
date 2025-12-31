/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <string.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <fstream>
#include "gstdsexample.h"
#include <sys/time.h>
GST_DEBUG_CATEGORY_STATIC (gst_dsexample_debug);
#define GST_CAT_DEFAULT gst_dsexample_debug
#define USE_EGLIMAGE 1
/* enable to write transformed cvmat to files */
/* #define DSEXAMPLE_DEBUG */
static GQuark _dsmeta_quark = 0;

/* Enum to identify properties */
enum
{
  PROP_0,
  PROP_UNIQUE_ID,
  PROP_PROCESSING_WIDTH,
  PROP_PROCESSING_HEIGHT,
  PROP_PROCESS_FULL_FRAME,
  PROP_BATCH_SIZE,
  PROP_BLUR_OBJECTS,
  PROP_GPU_DEVICE_ID,
    // VLM Queue Properties
  PROP_VLM_ENABLED,
  PROP_VLM_QUEUE_SIZE,
  PROP_VLM_FRAME_INTERVAL,
  PROP_VLM_SERVICE_URL
};

#define CHECK_NVDS_MEMORY_AND_GPUID(object, surface)  \
  ({ int _errtype=0;\
   do {  \
    if ((surface->memType == NVBUF_MEM_DEFAULT || surface->memType == NVBUF_MEM_CUDA_DEVICE) && \
        (surface->gpuId != object->gpu_id))  { \
    GST_ELEMENT_ERROR (object, RESOURCE, FAILED, \
        ("Input surface gpu-id doesnt match with configured gpu-id for element," \
         " please allocate input using unified memory, or use same gpu-ids"),\
        ("surface-gpu-id=%d,%s-gpu-id=%d",surface->gpuId,GST_ELEMENT_NAME(object),\
         object->gpu_id)); \
    _errtype = 1;\
    } \
    } while(0); \
    _errtype; \
  })


/* Default values for properties */
#define DEFAULT_UNIQUE_ID 15
#define DEFAULT_PROCESSING_WIDTH 640
#define DEFAULT_PROCESSING_HEIGHT 480
#define DEFAULT_PROCESS_FULL_FRAME TRUE
#define DEFAULT_BLUR_OBJECTS FALSE
#define DEFAULT_GPU_ID 0
#define DEFAULT_BATCH_SIZE 1

#define RGB_BYTES_PER_PIXEL 3
#define RGBA_BYTES_PER_PIXEL 4
#define Y_BYTES_PER_PIXEL 1
#define UV_BYTES_PER_PIXEL 2

#define MIN_INPUT_OBJECT_WIDTH 1
#define MIN_INPUT_OBJECT_HEIGHT 1

#define CHECK_NPP_STATUS(npp_status,error_str) do { \
  if ((npp_status) != NPP_SUCCESS) { \
    g_print ("Error: %s in %s at line %d: NPP Error %d\n", \
        error_str, __FILE__, __LINE__, npp_status); \
    goto error; \
  } \
} while (0)

#define CHECK_CUDA_STATUS(cuda_status,error_str) do { \
  if ((cuda_status) != cudaSuccess) { \
    g_print ("Error: %s in %s at line %d (%s)\n", \
        error_str, __FILE__, __LINE__, cudaGetErrorName(cuda_status)); \
    goto error; \
  } \
} while (0)

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_dsexample_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

static GstStaticPadTemplate gst_dsexample_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_dsexample_parent_class parent_class
G_DEFINE_TYPE (GstDsExample, gst_dsexample, GST_TYPE_BASE_TRANSFORM);

static void gst_dsexample_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_dsexample_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_dsexample_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_dsexample_start (GstBaseTransform * btrans);
static gboolean gst_dsexample_stop (GstBaseTransform * btrans);

static GstFlowReturn gst_dsexample_transform_ip (GstBaseTransform *
    btrans, GstBuffer * inbuf);

static std::shared_ptr<VLMFrameData> 
create_mock_frame_data(GstDsExample *dsexample, NvDsFrameMeta *frame_meta, guint batch_idx);

static void gst_dsexample_send_to_vlm_service(GstDsExample *dsexample,
    std::shared_ptr<struct VLMFrameData> frame_data);

static void gst_dsexample_vlm_worker (GstDsExample *dsexample);

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_dsexample_class_init (GstDsExampleClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  /* Indicates we want to use DS buf api */
  g_setenv ("DS_NEW_BUFAPI", "1", TRUE);

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  /* Overide base class functions */
  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_dsexample_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_dsexample_get_property);

  gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_dsexample_set_caps);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_dsexample_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_dsexample_stop);

  gstbasetransform_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_dsexample_transform_ip);

  /* Install properties */
  g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
      g_param_spec_uint ("unique-id",
          "Unique ID",
          "Unique ID for the element. Can be used to identify output of the"
          " element", 0, G_MAXUINT, DEFAULT_UNIQUE_ID, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESSING_WIDTH,
      g_param_spec_int ("processing-width",
          "Processing Width",
          "Width of the input buffer to algorithm",
          1, G_MAXINT, DEFAULT_PROCESSING_WIDTH, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESSING_HEIGHT,
      g_param_spec_int ("processing-height",
          "Processing Height",
          "Height of the input buffer to algorithm",
          1, G_MAXINT, DEFAULT_PROCESSING_HEIGHT, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESS_FULL_FRAME,
      g_param_spec_boolean ("full-frame",
          "Full frame",
          "Enable to process full frame or disable to process objects detected"
          "by primary detector", DEFAULT_PROCESS_FULL_FRAME, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_BLUR_OBJECTS,
      g_param_spec_boolean ("blur-objects",
          "Blur Objects",
          "Enable to blur the objects detected in full-frame=0 mode"
          "by primary detector", DEFAULT_BLUR_OBJECTS, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id",
          "Set GPU Device ID",
          "Set GPU Device ID", 0,
          G_MAXUINT, 0,
          GParamFlags
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_BATCH_SIZE,
      g_param_spec_uint ("batch-size", "Batch Size",
          "Maximum batch size for processing",
          1, NVDSEXAMPLE_MAX_BATCH_SIZE, DEFAULT_BATCH_SIZE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  // VLM Queue Properties
  g_object_class_install_property (gobject_class, PROP_VLM_ENABLED,
      g_param_spec_boolean ("vlm-enabled",
          "VLM Enabled",
          "Enable VLM frame processing queue", FALSE, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_VLM_QUEUE_SIZE,
      g_param_spec_uint ("vlm-queue-size",
          "VLM Queue Size",
          "Maximum size of VLM frame queue",
          1, 1000, 100, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_VLM_FRAME_INTERVAL,
      g_param_spec_uint ("vlm-frame-interval",
          "VLM Frame Interval",
          "Process every Nth frame for VLM (1 = every frame, 30 = every 30th frame)",
          1, 300, 30, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_VLM_SERVICE_URL,
      g_param_spec_string ("vlm-service-url",
          "VLM Service URL",
          "URL endpoint for VLM service",
          "http://localhost:8000/vlm/analyze", (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /* Set sink and src pad capabilities */
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_dsexample_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_dsexample_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class,
      "DsExample plugin",
      "DsExample Plugin",
      "Process a 3rdparty example algorithm on objects / full frame",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");
}

static void
gst_dsexample_init (GstDsExample * dsexample)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (dsexample);

  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), TRUE);

  /* Initialize all property variables to default values */
  dsexample->unique_id = DEFAULT_UNIQUE_ID;
  dsexample->processing_width = DEFAULT_PROCESSING_WIDTH;
  dsexample->processing_height = DEFAULT_PROCESSING_HEIGHT;
  dsexample->process_full_frame = DEFAULT_PROCESS_FULL_FRAME;
  dsexample->blur_objects = DEFAULT_BLUR_OBJECTS;
  dsexample->gpu_id = DEFAULT_GPU_ID;
  dsexample->max_batch_size = DEFAULT_BATCH_SIZE;

  // Initialize VLM queue and threading
  dsexample->vlm_enabled = TRUE;
  dsexample->vlm_thread_running = false;
  dsexample->vlm_frame_queue = std::make_shared<ThreadSafeQueue<VLMFrameData>>();

  dsexample->vlm_queue_max_size = 100;      // Maximum 100 frames in queue
  dsexample->vlm_frame_interval = 30;       // Process every 30th frame
  dsexample->vlm_frame_counter = 0;
  dsexample->vlm_service_url = g_strdup("http://localhost:8000/vlm/analyze");  // Default URL

  dsexample->redis_enabled = TRUE;
  dsexample->vlm_stream_manager = std::make_shared<VLMRedisStreamManager>("localhost", 6379);
    
  if (dsexample->vlm_stream_manager->is_connected()) {
      g_print("✅ VLM Redis Streams ready\n");
  } else {
      g_print("❌ VLM Redis Streams connection failed\n");
  }

  /* This quark is required to identify NvDsMeta when iterating through
   * the buffer metadatas */
  if (!_dsmeta_quark)
    _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_dsexample_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      dsexample->unique_id = g_value_get_uint (value);
      break;
    case PROP_PROCESSING_WIDTH:
      dsexample->processing_width = g_value_get_int (value);
      break;
    case PROP_PROCESSING_HEIGHT:
      dsexample->processing_height = g_value_get_int (value);
      break;
    case PROP_PROCESS_FULL_FRAME:
      dsexample->process_full_frame = g_value_get_boolean (value);
      break;
    case PROP_BLUR_OBJECTS:
      dsexample->blur_objects = g_value_get_boolean (value);
      break;
    case PROP_GPU_DEVICE_ID:
      dsexample->gpu_id = g_value_get_uint (value);
      break;
    case PROP_BATCH_SIZE:
      dsexample->max_batch_size = g_value_get_uint (value);
      break;
    case PROP_VLM_ENABLED:
      dsexample->vlm_enabled = g_value_get_boolean (value);
      break;
    case PROP_VLM_QUEUE_SIZE:
      dsexample->vlm_queue_max_size = g_value_get_uint (value);
      break;
    case PROP_VLM_FRAME_INTERVAL:
      dsexample->vlm_frame_interval = g_value_get_uint (value);
      break;
    case PROP_VLM_SERVICE_URL:
      if (dsexample->vlm_service_url) {
        g_free (dsexample->vlm_service_url);
      }
      dsexample->vlm_service_url = g_value_dup_string (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void
gst_dsexample_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (object);

  switch (prop_id) {
    case PROP_UNIQUE_ID:
      g_value_set_uint (value, dsexample->unique_id);
      break;
    case PROP_PROCESSING_WIDTH:
      g_value_set_int (value, dsexample->processing_width);
      break;
    case PROP_PROCESSING_HEIGHT:
      g_value_set_int (value, dsexample->processing_height);
      break;
    case PROP_PROCESS_FULL_FRAME:
      g_value_set_boolean (value, dsexample->process_full_frame);
      break;
    case PROP_BLUR_OBJECTS:
      g_value_set_boolean (value, dsexample->blur_objects);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, dsexample->gpu_id);
      break;
    case PROP_BATCH_SIZE:
      g_value_set_uint (value, dsexample->max_batch_size);
      break;
    case PROP_VLM_ENABLED:
      g_value_set_boolean (value, dsexample->vlm_enabled);
      break;
    case PROP_VLM_QUEUE_SIZE:
      g_value_set_uint (value, dsexample->vlm_queue_max_size);
      break;
    case PROP_VLM_FRAME_INTERVAL:
      g_value_set_uint (value, dsexample->vlm_frame_interval);
      break;
    case PROP_VLM_SERVICE_URL:
      g_value_set_string (value, dsexample->vlm_service_url);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * Initialize all resources and start the output thread
 */
static gboolean
gst_dsexample_start (GstBaseTransform * btrans)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (btrans);
  NvBufSurfaceCreateParams create_params = { 0 };
  DsExampleInitParams init_params =
      { dsexample->processing_width, dsexample->processing_height,
    dsexample->process_full_frame
  };

  int val = -1;

  /* Algorithm specific initializations and resource allocation. */
  dsexample->dsexamplelib_ctx = DsExampleCtxInit (&init_params);

  GST_DEBUG_OBJECT (dsexample, "ctx lib %p \n", dsexample->dsexamplelib_ctx);

  CHECK_CUDA_STATUS (cudaSetDevice (dsexample->gpu_id),
      "Unable to set cuda device");

  cudaDeviceGetAttribute (&val, cudaDevAttrIntegrated, dsexample->gpu_id);
  dsexample->is_integrated = val;

  GST_DEBUG_OBJECT (dsexample, "Setting batch-size %d \n",
      dsexample->max_batch_size);

  if (dsexample->process_full_frame && dsexample->blur_objects) {
    GST_ERROR ("Error: does not support blurring while processing full frame");
    goto error;
  }

#ifndef WITH_OPENCV
  if (dsexample->blur_objects) {
    GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
          ("OpenCV has been deprecated, hence object blurring will not work."
          "Enable OpenCV compilation in gst-dsexample Makefile by setting 'WITH_OPENCV:=1"), (NULL));
    goto error;
  }
#endif

  CHECK_CUDA_STATUS (cudaStreamCreate (&dsexample->cuda_stream),
      "Could not create cuda stream");

  if (dsexample->inter_buf)
    NvBufSurfaceDestroy (dsexample->inter_buf);
  dsexample->inter_buf = NULL;

  /* An intermediate buffer for NV12/RGBA to BGR conversion  will be
   * required. Can be skipped if custom algorithm can work directly on NV12/RGBA. */
  create_params.gpuId  = dsexample->gpu_id;
  create_params.width  = dsexample->processing_width;
  create_params.height = dsexample->processing_height;
  create_params.size = 0;
  create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
  create_params.layout = NVBUF_LAYOUT_PITCH;

  if(dsexample->is_integrated) {
    create_params.memType = NVBUF_MEM_DEFAULT;
  }
  else {
    create_params.memType = NVBUF_MEM_CUDA_PINNED;
  }

  if (NvBufSurfaceCreate (&dsexample->inter_buf, 1,
          &create_params) != 0) {
    GST_ERROR ("Error: Could not allocate internal buffer for dsexample");
    goto error;
  }

  /* Create host memory for storing converted/scaled interleaved RGB data */
  CHECK_CUDA_STATUS (cudaMallocHost (&dsexample->host_rgb_buf,
          dsexample->processing_width * dsexample->processing_height *
          RGB_BYTES_PER_PIXEL), "Could not allocate cuda host buffer");

  GST_DEBUG_OBJECT (dsexample, "allocated cuda buffer %p \n",
      dsexample->host_rgb_buf);

#ifdef WITH_OPENCV
  /* CV Mat containing interleaved RGB data. This call does not allocate memory.
   * It uses host_rgb_buf as data. */
  dsexample->cvmat =
      new cv::Mat (dsexample->processing_height, dsexample->processing_width,
      CV_8UC3, dsexample->host_rgb_buf,
      dsexample->processing_width * RGB_BYTES_PER_PIXEL);

  if (!dsexample->cvmat)
    goto error;

  GST_DEBUG_OBJECT (dsexample, "created CV Mat\n");
#endif

  /* Set the NvBufSurfTransform config parameters. */
  dsexample->transform_config_params.compute_mode =
      NvBufSurfTransformCompute_Default;
  dsexample->transform_config_params.gpu_id = dsexample->gpu_id;
  

  // Start VLM worker thread
  if (dsexample->vlm_enabled) {
    dsexample->vlm_thread_running = true;
    dsexample->vlm_worker_thread = std::thread(gst_dsexample_vlm_worker, dsexample);
  }

  return TRUE;
error:
  if (dsexample->host_rgb_buf) {
    cudaFreeHost (dsexample->host_rgb_buf);
    dsexample->host_rgb_buf = NULL;
  }

  if (dsexample->cuda_stream) {
    cudaStreamDestroy (dsexample->cuda_stream);
    dsexample->cuda_stream = NULL;
  }
  if (dsexample->dsexamplelib_ctx)
    DsExampleCtxDeinit (dsexample->dsexamplelib_ctx);
  return FALSE;
}

/**
 * Stop the output thread and free up all the resources
 */
static gboolean
gst_dsexample_stop (GstBaseTransform * btrans)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (btrans);

  // ✅ FIX: Stop VLM worker thread FIRST
  if (dsexample->vlm_enabled && dsexample->vlm_thread_running) {
    g_print("Stopping VLM worker thread...\n");
    
    dsexample->vlm_thread_running = false;
    dsexample->vlm_frame_queue->terminate();  // ✅ Safe shutdown!
    
    if (dsexample->vlm_worker_thread.joinable()) {
      dsexample->vlm_worker_thread.join();
    }
  }

  if (dsexample->inter_buf)
    NvBufSurfaceDestroy(dsexample->inter_buf);
  dsexample->inter_buf = NULL;

  if (dsexample->cuda_stream)
    cudaStreamDestroy (dsexample->cuda_stream);
  dsexample->cuda_stream = NULL;

#ifdef WITH_OPENCV
  delete dsexample->cvmat;
  dsexample->cvmat = NULL;
#endif

  if (dsexample->host_rgb_buf) {
    cudaFreeHost (dsexample->host_rgb_buf);
    dsexample->host_rgb_buf = NULL;
  }

  GST_DEBUG_OBJECT (dsexample, "deleted CV Mat \n");

  /* Deinit the algorithm library */
  DsExampleCtxDeinit (dsexample->dsexamplelib_ctx);
  dsexample->dsexamplelib_ctx = NULL;

  GST_DEBUG_OBJECT (dsexample, "ctx lib released \n");

  return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_dsexample_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (btrans);
  /* Save the input video information, since this will be required later. */
  gst_video_info_from_caps (&dsexample->video_info, incaps);

  if (dsexample->blur_objects && !dsexample->process_full_frame) {
    /* requires RGBA format for blurring the objects in opencv */
     if (dsexample->video_info.finfo->format != GST_VIDEO_FORMAT_RGBA) {
      GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
          ("input format should be RGBA when using blur-objects property"), (NULL));
      goto error;
      }
  }

  return TRUE;

error:
  return FALSE;
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_dsexample_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf)
{
  GstDsExample *dsexample = GST_DSEXAMPLE (btrans);
  GstMapInfo in_map_info;
  GstFlowReturn flow_ret = GST_FLOW_ERROR;

  NvBufSurface *surface = NULL;
  NvDsBatchMeta *batch_meta = NULL;
  NvDsFrameMeta *frame_meta = NULL;
  NvDsMetaList * l_frame = NULL;

  dsexample->frame_num++;
  CHECK_CUDA_STATUS (cudaSetDevice (dsexample->gpu_id),
      "Unable to set cuda device");

  memset (&in_map_info, 0, sizeof (in_map_info));
  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    g_print ("Error: Failed to map gst buffer\n");
    goto error;
  }

  nvds_set_input_system_timestamp (inbuf, GST_ELEMENT_NAME (dsexample));
  surface = (NvBufSurface *) in_map_info.data;
  GST_DEBUG_OBJECT (dsexample,
      "Processing Frame %" G_GUINT64_FORMAT " Surface %p\n",
      dsexample->frame_num, surface);

  if (CHECK_NVDS_MEMORY_AND_GPUID (dsexample, surface))
    goto error;

  batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (batch_meta == nullptr) {
    GST_ELEMENT_ERROR (dsexample, STREAM, FAILED,
        ("NvDsBatchMeta not found for input buffer."), (NULL));
    return GST_FLOW_ERROR;
  }

  if (dsexample->vlm_enabled) {
    NvDsFrameMeta *frame_meta = NULL;
    NvDsMetaList *l_frame = NULL;
    NvDsObjectMeta *obj_meta = NULL;
    NvDsMetaList *l_obj = NULL;
    guint frame_index = 0;

    // Increment global counter once per batch
    // Rate limiting: only process every Nth frame
    dsexample->vlm_frame_counter++;
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
      frame_meta = (NvDsFrameMeta *) (l_frame->data);
      frame_index = frame_meta->frame_num;
      
      if (dsexample->vlm_frame_counter % dsexample->vlm_frame_interval == 0) {
        // Create mock frame data (no actual frame extraction)
        VLMFrameData vlm_frame;
        vlm_frame.width = 1920;
        vlm_frame.height = 1080;
        vlm_frame.timestamp = frame_meta->buf_pts;
        vlm_frame.source_id = frame_meta->source_id;
        vlm_frame.frame_number = frame_meta->frame_num;
        vlm_frame.format = "RGB";
        
        if (dsexample->vlm_frame_queue->size() >= dsexample->vlm_queue_max_size) {
          VLMFrameData dropped;
          dsexample->vlm_frame_queue->try_pop(dropped);  // Drop oldest
        }

        // Thread-safe push (no manual locking!)
        dsexample->vlm_frame_queue->push(std::move(vlm_frame));
        g_print ("Source_id=%d enqueued frame #%d to VLM queue (size=%u)\n", 
          frame_meta->source_id, frame_index, dsexample->vlm_frame_queue->size());    
      }
    }  
  }

  flow_ret = GST_FLOW_OK;

error:

  nvds_set_output_system_timestamp (inbuf, GST_ELEMENT_NAME (dsexample));
  gst_buffer_unmap (inbuf, &in_map_info);
  return flow_ret;
}

static void
gst_dsexample_vlm_worker (GstDsExample *dsexample)
{
  GST_INFO_OBJECT (dsexample, "VLM worker thread started");
  uint32_t processed_count = 0;
  
  while (dsexample->vlm_thread_running) {
    std::shared_ptr<VLMFrameData> frame_data = nullptr;
    
    auto frame_data_ptr = dsexample->vlm_frame_queue->wait_and_pop();
    
    if (!frame_data_ptr || !dsexample->vlm_thread_running) {
      break;  // Queue terminated or shutdown requested
    }
    
    gst_dsexample_send_to_vlm_service(dsexample, frame_data_ptr);
    processed_count++;
  }
  
  GST_INFO_OBJECT (dsexample, "VLM worker thread stopped after processing %u frames", processed_count);
}

static void
gst_dsexample_send_to_vlm_service(GstDsExample *dsexample, 
                                  std::shared_ptr<VLMFrameData> frame_data)
{
  // Example placeholder:
  try {
    if (!frame_data) {
      g_print ("Invalid frame_data pointer");
      return;
    }

    // std::string vlm_response = call_vlm_service(dsexample->vlm_service_url, frame_data);

    std::string vlm_response = 
        "{ \"description\": \"A person riding a horse on a beach.\", "
        "\"objects\": [ {\"label\": \"person\", \"confidence\": 0.98}, "
        "{\"label\": \"horse\", \"confidence\": 0.95}, "
        "{\"label\": \"beach\", \"confidence\": 0.90} ] }";

    // Add to Redis stream
    if (dsexample->redis_enabled && dsexample->vlm_stream_manager) {
        std::string msg_id = dsexample->vlm_stream_manager->add_vlm_result(
            frame_data->frame_number, 
            frame_data->source_id, 
            vlm_response,
            "deepstream_vlm_v1"  // Model version
        );
        
        g_print("VLM result added to stream: %s\n", msg_id.c_str());
    }
    
  } catch (const std::exception& e) {
    GST_ERROR_OBJECT (dsexample, "VLM service error: %s", e.what());
  }
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
dsexample_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_dsexample_debug, "dsexample", 0,
      "dsexample plugin");

  return gst_element_register (plugin, "dsexample", GST_RANK_PRIMARY,
      GST_TYPE_DSEXAMPLE);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_dsexample,
    DESCRIPTION, dsexample_plugin_init, "8.0", LICENSE, BINARY_PACKAGE, URL)
