/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * Modified DeepStream Test3 with demuxer support
 * TILER_DISPLAY=1: Standard tiler mode (default)
 * TILER_DISPLAY=0: streammux -> pgie -> demuxer -> individual fakesinks
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>

#include "gstnvdsmeta.h"
#include "nvds_yml_parser.h"
#include "gst-nvmessage.h"
#include "deepstream_common.h"

#define MAX_DISPLAY_LEN 64
#define MAX_SOURCES 16

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/* By default, OSD process-mode is set to GPU_MODE. To change mode, set as:
 * 0: CPU mode
 * 1: GPU mode
 */
#define OSD_PROCESS_MODE 1

/* By default, OSD will not display text. To display text, change this to 1 */
#define OSD_DISPLAY_TEXT 0

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

#define TILED_OUTPUT_WIDTH 1280
#define TILED_OUTPUT_HEIGHT 720

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

/* Check for parsing error. */
#define RETURN_ON_PARSER_ERROR(parse_expr) \
  if (NVDS_YAML_PARSER_SUCCESS != parse_expr) { \
    g_printerr("Error in parsing configuration file.\n"); \
    return -1; \
  }

gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
  "RoadSign"
};

static gboolean PERF_MODE = FALSE;
static gboolean TILER_DISPLAY_MODE = FALSE;
static gboolean DEMUXER_MODE = FALSE;

/* Structure to hold per-stream sink information */
typedef struct {
  GstElement *dsexample;
  GstElement *fakesink;
  guint stream_id;
} StreamSink;

/* Global array to hold individual stream sinks */
static StreamSink stream_sinks[MAX_SOURCES];
static guint num_stream_sinks = 0;

/* tiler_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
tiler_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0; 
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        int offset = 0;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
                l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
            if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
                vehicle_count++;
                num_rects++;
            }
            if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
                person_count++;
                num_rects++;
            }
        }
        
        if (TILER_DISPLAY_MODE) {
          g_print ("Frame Number = %d Number of objects = %d "
            "Vehicle Count = %d Person Count = %d\n",
            frame_meta->frame_num, num_rects, vehicle_count, person_count);
        } else {
          g_print ("Stream %d Frame %d: Objects=%d Vehicle=%d Person=%d\n",
            frame_meta->source_id, frame_meta->frame_num, num_rects, vehicle_count, person_count);
        }
        
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        NvOSD_TextParams *txt_params  = &display_meta->text_params;
        txt_params->display_text = g_malloc0 (MAX_DISPLAY_LEN);
        offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ", person_count);
        offset = snprintf(txt_params->display_text + offset , MAX_DISPLAY_LEN, "Vehicle = %d ", vehicle_count);

        /* Now set the offsets where the string should appear */
        txt_params->x_offset = 10;
        txt_params->y_offset = 12;

        /* Font , font-color and font-size */
        txt_params->font_params.font_name = "Serif";
        txt_params->font_params.font_size = 10;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;

        /* Text background color */
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;

        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }
    return GST_PAD_PROBE_OK;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_WARNING:
    {
      gchar *debug = NULL;
      GError *error = NULL;
      gst_message_parse_warning (msg, &error, &debug);
      g_printerr ("WARNING from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      g_free (debug);
      g_printerr ("Warning: %s\n", error->message);
      g_error_free (error);
      break;
    }
    case GST_MESSAGE_ERROR:
    {
      gchar *debug = NULL;
      GError *error = NULL;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    case GST_MESSAGE_ELEMENT:
    {
      if (gst_nvmessage_is_stream_eos (msg)) {
        guint stream_id = 0;
        if (gst_nvmessage_parse_stream_eos (msg, &stream_id)) {
          g_print ("Got EOS from stream %d - stopping pipeline\n", stream_id);
          g_main_loop_quit (loop);
        }
      }
      break;
    }
    default:
      break;
  }
  return TRUE;
}

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  if (!caps) {
    caps = gst_pad_query_caps (decoder_src_pad, NULL);
  }
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
     * NVMM memory features. */
    if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
      /* Get the source bin ghost pad */
      GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
      if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
              decoder_src_pad)) {
        g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref (bin_ghost_pad);
    } else {
      g_printerr ("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
  
  gst_caps_unref (caps);
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  g_print ("Decodebin child added: %s\n", name);
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
  if (g_strrstr (name, "source") == name) {
    g_object_set(G_OBJECT(object), "drop-on-latency", TRUE, NULL);
  }
}

static GstElement *
create_source_bin (guint index, gchar * uri)
{
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = { };

  g_snprintf (bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new (bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  if (PERF_MODE) {
    uri_decode_bin = gst_element_factory_make ("nvurisrcbin", "uri-decode-bin");
    g_object_set (G_OBJECT (uri_decode_bin), "file-loop", TRUE, NULL);
    g_object_set (G_OBJECT (uri_decode_bin), "cudadec-memtype", 0, NULL);
  } else {
    uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");
  }

  if (!bin || !uri_decode_bin) {
    g_printerr ("One element in source bin could not be created.\n");
    return NULL;
  }

  /* We set the input uri to the source element */
  g_object_set (G_OBJECT (uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), bin);
  g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);

  gst_bin_add (GST_BIN (bin), uri_decode_bin);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}

/* Create individual fakesink for stream */
static gboolean
create_stream_sink (guint stream_id, GstElement *pipeline)
{
  StreamSink *sink_info = &stream_sinks[stream_id];
  gchar element_name[64];
  
  sink_info->stream_id = stream_id;

  // Create dsexample for this stream
  g_snprintf(element_name, sizeof(element_name), "dsexample-%d", stream_id);
  sink_info->dsexample = gst_element_factory_make("dsexample", element_name);
  
  if (!sink_info->dsexample) {
    g_printerr("Failed to create dsexample for stream %d\n", stream_id);
    return FALSE;
  }

  // Configure dsexample properties
  g_object_set(G_OBJECT(sink_info->dsexample),
                "unique-id", stream_id + 15,        // Unique ID per stream
                "vlm-enabled", TRUE,                // Enable VLM for each stream
                "vlm-queue-size", 10,               // Individual queue per stream
                "vlm-frame-interval", 30,           // Process every 30th frame
                "processing-width", 640,
                "processing-height", 480,
                NULL);
  
  /* Create fakesink for this stream */
  g_snprintf(element_name, sizeof(element_name), "fakesink-%d", stream_id);
  sink_info->fakesink = gst_element_factory_make("fakesink", element_name);
  
  if (!sink_info->fakesink) {
    g_printerr("Failed to create fakesink for stream %d\n", stream_id);
    return FALSE;
  }

  /* Configure fakesink */
  g_object_set(G_OBJECT(sink_info->fakesink),
               "sync", TRUE,             /* Enable sync for timing */
               "async", FALSE,
               "silent", FALSE,
               NULL);

  /* Add fakesink to pipeline */
  gst_bin_add_many(GST_BIN(pipeline), sink_info->dsexample, sink_info->fakesink, NULL);
  // Link dsexample -> fakesink
  if (!gst_element_link(sink_info->dsexample, sink_info->fakesink)) {
      g_printerr("Failed to link dsexample to fakesink for stream %d\n", stream_id);
      gst_object_unref(sink_info->dsexample);
      gst_object_unref(sink_info->fakesink);
      return FALSE;
  }

  g_print("Created stream %d: demux -> dsexample -> fakesink\n", stream_id);
  num_stream_sinks++;
  return TRUE;
}

/* Link demuxer output to individual fakesinks */
static void
demuxer_pad_added (GstElement *demux, GstPad *new_pad, gpointer user_data)
{
  GstElement *pipeline = (GstElement *) user_data;
  gchar *pad_name = gst_pad_get_name(new_pad);
  guint stream_id = 0;
  
  /* Extract stream ID from pad name (format: src_X) */
  if (sscanf(pad_name, "src_%u", &stream_id) == 1) {
    if (stream_id < num_stream_sinks) {
      StreamSink *sink_info = &stream_sinks[stream_id];
      GstPad *dsexample_sink_pad = gst_element_get_static_pad(sink_info->dsexample, "sink");
      
      if (gst_pad_link(new_pad, dsexample_sink_pad) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link demuxer pad to stream %d dsexample\n", stream_id);
      } else {
        g_print("Linked demuxer output to dsexample for stream %d\n", stream_id);
      }
      
      gst_object_unref(dsexample_sink_pad);
    }
  }
  
  g_free(pad_name);
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL,
      *queue1, *queue2, *queue3, *queue4, *queue5, *nvvidconv = NULL,
      *nvosd = NULL, *tiler = NULL, *nvdslogger = NULL, *streamdemux = NULL, *dsexample = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *tiler_src_pad = NULL;
  guint i = 0, num_sources = 0;
  guint tiler_rows, tiler_columns;
  guint pgie_batch_size;
  gboolean yaml_config = FALSE;
  NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;
  
  /* Check environment variables */
  PERF_MODE = g_getenv("NVDS_TEST3_PERF_MODE") &&
      !g_strcmp0(g_getenv("NVDS_TEST3_PERF_MODE"), "1");

  if (g_getenv("TILER_DISPLAY") && !g_strcmp0(g_getenv("TILER_DISPLAY"), "0")) {
    TILER_DISPLAY_MODE = FALSE;
    g_print("Tiler display disabled - using stream demux mode\n");
  }

  g_print ("PERF_MODE : %s\n", PERF_MODE ? "ON" : "OFF");
  g_print ("TILER_DISPLAY_MODE : %s\n", TILER_DISPLAY_MODE ? "ON" : "OFF");
  g_print ("DEMUXER_MODE : %s\n", DEMUXER_MODE ? "ON" : "OFF");
  
  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  /* Check input arguments */
  if (argc < 2) {
    g_printerr ("Usage: %s <yml file>\n", argv[0]);
    g_printerr ("OR: %s <uri1> [uri2] ... [uriN] \n", argv[0]);
    g_printerr ("\nEnvironment variables:\n");
    g_printerr ("  TILER_DISPLAY=0         - Use stream demux + individual fakesinks\n");
    g_printerr ("  TILER_DISPLAY=1         - Use tiler display (default)\n");
    g_printerr ("  NVDS_TEST3_PERF_MODE=1  - Enable performance mode\n");
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Parse inference plugin type */
  yaml_config = (g_str_has_suffix (argv[1], ".yml") ||
          g_str_has_suffix (argv[1], ".yaml"));

  if (yaml_config) {
    RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&pgie_type, argv[1],
                "primary-gie"));
  }

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dstest3-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add (GST_BIN (pipeline), streammux);

  GList *src_list = NULL;

  if (yaml_config) {
    RETURN_ON_PARSER_ERROR(nvds_parse_source_list(&src_list, argv[1], "source-list"));

    GList * temp = src_list;
    while(temp) {
      num_sources++;
      temp=temp->next;
    }
    g_list_free(temp);
  } else {
      num_sources = argc - 1;
  }

  for (i = 0; i < num_sources; i++) {
    GstPad *sinkpad, *srcpad;
    gchar pad_name[16] = { };

    GstElement *source_bin = NULL;
    if (g_str_has_suffix (argv[1], ".yml") || g_str_has_suffix (argv[1], ".yaml")) {
      g_print("Now playing : %s\n",(char*)(src_list)->data);
      source_bin = create_source_bin (i, (char*)(src_list)->data);
    } else {
      source_bin = create_source_bin (i, argv[i + 1]);
    }
    if (!source_bin) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }

    gst_bin_add (GST_BIN (pipeline), source_bin);

    g_snprintf (pad_name, 15, "sink_%u", i);
    sinkpad = gst_element_request_pad_simple (streammux, pad_name);
    if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad (source_bin, "src");
    if (!srcpad) {
      g_printerr ("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref (srcpad);
    gst_object_unref (sinkpad);

    if (yaml_config) {
      src_list = src_list->next;
    }
  }

  if (yaml_config) {
    g_list_free(src_list);
  }

  /* Use nvinfer or nvinferserver to infer on batched frame. */
  if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
    pgie = gst_element_factory_make ("nvinferserver", "primary-nvinference-engine");
  } else {
    pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
  }

  /* Add queue elements between every two elements */
  queue1 = gst_element_factory_make ("queue", "queue1");
  queue2 = gst_element_factory_make ("queue", "queue2");
  queue3 = gst_element_factory_make ("queue", "queue3");
  queue4 = gst_element_factory_make ("queue", "queue4");
  queue5 = gst_element_factory_make ("queue", "queue5");

  /* Use nvdslogger for perf measurement. */
  nvdslogger = gst_element_factory_make ("nvdslogger", "nvdslogger");

  /* Create elements based on display mode */
  if (TILER_DISPLAY_MODE) {
    /* Tiler mode elements */
    tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");
    nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
    nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");
    
    if (PERF_MODE) {
      sink = gst_element_factory_make ("fakesink", "nvvideo-renderer");
    } else {
      /* Finally render the osd output */
      if(prop.integrated) {
        sink = gst_element_factory_make ("nv3dsink", "nv3d-sink");
      } else {
        if (TILER_DISPLAY_MODE) {
#ifdef __aarch64__
          sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
#else
          sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
#endif
        } else {
          sink = gst_element_factory_make ("fakesink", "nvvideo-renderer");
        }
      }
    }
    
    if (!tiler || !nvvidconv || !nvosd || !sink) {
      g_printerr ("Tiler mode elements could not be created. Exiting.\n");
      return -1;
    }
  } else {

    if (DEMUXER_MODE) {
      /* Demux mode elements */
      streamdemux = gst_element_factory_make ("nvstreamdemux", "stream-demuxer");
      if (!streamdemux) {
        g_printerr ("Stream demux element could not be created. Exiting.\n");
        return -1;
      }
      
      gst_bin_add(GST_BIN(pipeline), streamdemux);
      /* Create individual fakesinks for each stream */
      g_printf("Creating individual fakesinks for %d streams\n", num_sources);
      for (i = 0; i < num_sources; i++) {
        if (!create_stream_sink(i, pipeline)) {
          g_printerr("Failed to create stream sink for stream %d. Exiting.\n", i);
          return -1;
        }
      }
      
      /* Connect demuxer pad-added signal */
      // g_signal_connect(streamdemux, "pad-added", G_CALLBACK(demuxer_pad_added), pipeline);
      for (i = 0; i < num_sources; i++) {
        StreamSink *sink_info = &stream_sinks[i];
        if (!link_element_to_demux_src_pad(streamdemux, sink_info->dsexample, i)) {
          g_printerr("Failed to link demuxer to dsexample for stream %d. Exiting.\n", i);
          return -1;
        }
      }

    } else {
      dsexample = gst_element_factory_make("dsexample", "dsexample");
      sink = gst_element_factory_make ("fakesink", "nvvideo-renderer");
    }
  }

  if (!pgie || !nvdslogger) {
    g_printerr ("Basic elements could not be created. Exiting.\n");
    return -1;
  }

  /* Configure elements */
  if (yaml_config) {
    RETURN_ON_PARSER_ERROR(nvds_parse_streammux(streammux, argv[1],"streammux"));
    RETURN_ON_PARSER_ERROR(nvds_parse_gie(pgie, argv[1], "primary-gie"));

    g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
    if (pgie_batch_size != num_sources) {
      g_printerr
          ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
          pgie_batch_size, num_sources);
      g_object_set (G_OBJECT (pgie), "batch-size", num_sources, NULL);
    }

    if (TILER_DISPLAY_MODE) {
      RETURN_ON_PARSER_ERROR(nvds_parse_osd(nvosd, argv[1],"osd"));

      tiler_rows = (guint) sqrt (num_sources);
      tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
      g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns, NULL);

      RETURN_ON_PARSER_ERROR(nvds_parse_tiler(tiler, argv[1], "tiler"));
    
      if (PERF_MODE) {
        RETURN_ON_PARSER_ERROR(nvds_parse_fake_sink (sink, argv[1], "sink"));
      } else if(prop.integrated) {
        RETURN_ON_PARSER_ERROR(nvds_parse_3d_sink(sink, argv[1], "sink"));
      } else {
        if (TILER_DISPLAY_MODE) {
#ifdef __aarch64__
          RETURN_ON_PARSER_ERROR(nvds_parse_3d_sink(sink, argv[1], "sink"));
#else
          RETURN_ON_PARSER_ERROR(nvds_parse_egl_sink(sink, argv[1], "sink"));
#endif
        }
        else {
          RETURN_ON_PARSER_ERROR(nvds_parse_fake_sink(sink, argv[1], "sink"));
        }
      }
    }
  } else {
    g_object_set (G_OBJECT (streammux), "batch-size", num_sources, NULL);
    g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
        MUXER_OUTPUT_HEIGHT,
        "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

    /* Configure the nvinfer element using the nvinfer config file. */
    g_object_set (G_OBJECT (pgie),
        "config-file-path", "dstest3_pgie_config.txt", NULL);

    /* Override the batch-size set in the config file with the number of sources. */
    g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
    if (pgie_batch_size != num_sources) {
      g_printerr
          ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
          pgie_batch_size, num_sources);
      g_object_set (G_OBJECT (pgie), "batch-size", num_sources, NULL);
    }

    if (TILER_DISPLAY_MODE) {
      tiler_rows = (guint) sqrt (num_sources);
      tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
      /* we set the tiler properties here */
      g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns,
          "width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT, NULL);

      g_object_set (G_OBJECT (nvosd), "process-mode", OSD_PROCESS_MODE,
          "display-text", OSD_DISPLAY_TEXT, NULL);

      g_object_set (G_OBJECT (sink), "qos", 0, NULL);
    }
  }

  if (PERF_MODE) {
      if(prop.integrated) {
          g_object_set (G_OBJECT (streammux), "nvbuf-memory-type", 4, NULL);
      } else {
          g_object_set (G_OBJECT (streammux), "nvbuf-memory-type", 2, NULL);
      }
  }

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  if (TILER_DISPLAY_MODE) {
    /* Tiler mode: streammux -> pgie -> nvdslogger -> tiler -> nvvidconv -> nvosd -> sink */
    gst_bin_add_many (GST_BIN (pipeline), queue1, pgie, queue2, nvdslogger, tiler,
        queue3, nvvidconv, queue4, nvosd, queue5, sink, NULL);
    
    if (!gst_element_link_many (streammux, queue1, pgie, queue2, nvdslogger, tiler,
          queue3, nvvidconv, queue4, nvosd, queue5, sink, NULL)) {
      g_printerr ("Tiler pipeline elements could not be linked. Exiting.\n");
      return -1;
    }
  } else {
    /* Demux mode: streammux -> pgie -> nvdslogger -> streamdemux -> individual fakesinks */
    gst_bin_add_many (GST_BIN (pipeline), queue1, pgie, queue2, nvdslogger, NULL);
    if (!DEMUXER_MODE) {
      gst_bin_add_many (GST_BIN (pipeline), dsexample, sink, NULL);
    }
    
    if (DEMUXER_MODE) {
      if (!gst_element_link_many (streammux, queue1, pgie, queue2, nvdslogger, streamdemux, NULL)) {
        g_printerr ("Demux pipeline elements could not be linked. Exiting.\n");
        return -1;
      }
    } else {
      if (!gst_element_link_many (streammux, queue1, pgie, queue2, nvdslogger, dsexample, sink, NULL)) {
        g_printerr ("Pipeline elements could not be linked. Exiting.\n");
        return -1;
      }
    }
    
  }

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  tiler_src_pad = gst_element_get_static_pad (pgie, "src");
  if (!tiler_src_pad)
    g_print ("Unable to get src pad\n");
  else
    gst_pad_add_probe (tiler_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        tiler_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref (tiler_src_pad);

  /* Print configuration summary */
  g_print ("\n=== Pipeline Configuration ===\n");
  g_print ("Performance Mode: %s\n", PERF_MODE ? "Enabled" : "Disabled");
  g_print ("Tiler Display: %s\n", TILER_DISPLAY_MODE ? "Enabled" : "Disabled");
  g_print ("Number of Sources: %d\n", num_sources);
  
  if (!TILER_DISPLAY_MODE) {
    g_print ("Demux mode: streammux -> pgie -> nvdslogger -> streamdemux -> %d fakesinks\n", num_stream_sinks);
  }

  /* Set the pipeline to "playing" state */
  if (yaml_config) {
    g_print ("Using file: %s\n", argv[1]);
  }
  else {
    g_print ("Now playing:");
    for (i = 0; i < num_sources; i++) {
      g_print (" %s,", argv[i + 1]);
    }
    g_print ("\n");
  }
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}