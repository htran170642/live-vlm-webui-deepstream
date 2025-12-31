// RedisClient.h - Redis client with Streams support for VLM DeepStream project
#ifndef REDIS_CLIENT_H
#define REDIS_CLIENT_H

#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <iostream>
#include <chrono>
#include <map>
#include <hiredis/hiredis.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Redis Stream message structure
struct StreamMessage {
    std::string id;           // Redis stream ID (e.g., "1672531200000-0")
    std::map<std::string, std::string> fields;  // Key-value pairs
    uint64_t timestamp;       // Parsed timestamp from ID
    
    StreamMessage() : timestamp(0) {}
    
    StreamMessage(const std::string& stream_id, const std::map<std::string, std::string>& data) 
        : id(stream_id), fields(data) {
        // Parse timestamp from stream ID (format: timestamp-sequence)
        size_t dash_pos = stream_id.find('-');
        if (dash_pos != std::string::npos) {
            timestamp = std::stoull(stream_id.substr(0, dash_pos));
        } else {
            timestamp = 0;
        }
    }
    
    // Helper to get field as string
    std::string get_field(const std::string& key, const std::string& default_value = "") const {
        auto it = fields.find(key);
        return (it != fields.end()) ? it->second : default_value;
    }
    
    // Helper to get field as number
    template<typename T>
    T get_field_as(const std::string& key, T default_value = T{}) const {
        auto it = fields.find(key);
        if (it != fields.end()) {
            if constexpr (std::is_same_v<T, int> || std::is_same_v<T, uint32_t>) {
                return static_cast<T>(std::stoi(it->second));
            } else if constexpr (std::is_same_v<T, uint64_t>) {
                return std::stoull(it->second);
            } else if constexpr (std::is_same_v<T, double>) {
                return std::stod(it->second);
            }
        }
        return default_value;
    }
};

class RedisClient {
public:
    RedisClient(const std::string& host = "localhost", int port = 6379, const std::string& password = "")
        : host_(host), port_(port), password_(password), context_(nullptr), connected_(false) {}
    
    ~RedisClient() {
        disconnect();
    }

    // Connection management (same as before)
    bool connect() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (connected_) return true;
        
        context_ = redisConnect(host_.c_str(), port_);
        if (context_ == nullptr || context_->err) {
            if (context_) {
                std::cerr << "Redis connection error: " << context_->errstr << std::endl;
                redisFree(context_);
                context_ = nullptr;
            }
            return false;
        }
        
        if (!password_.empty()) {
            redisReply* reply = (redisReply*)redisCommand(context_, "AUTH %s", password_.c_str());
            if (reply->type == REDIS_REPLY_ERROR) {
                std::cerr << "Redis auth error: " << reply->str << std::endl;
                freeReplyObject(reply);
                redisFree(context_);
                context_ = nullptr;
                return false;
            }
            freeReplyObject(reply);
        }
        
        connected_ = true;
        std::cout << "✅ Redis connected with Streams support: " << host_ << ":" << port_ << std::endl;
        return true;
    }
    
    void disconnect() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (context_) {
            redisFree(context_);
            context_ = nullptr;
        }
        connected_ = false;
    }
    
    bool is_connected() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return connected_ && context_ != nullptr;
    }

    // ✅ NEW: Redis Streams operations
    
    // Add message to stream with auto-generated ID
    std::string xadd(const std::string& stream_key, const std::map<std::string, std::string>& fields) {
        if (!ensure_connected()) return "";
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Build XADD command: XADD stream_key * field1 value1 field2 value2 ...
        std::vector<const char*> argv;
        std::vector<std::string> args;
        
        args.push_back("XADD");
        args.push_back(stream_key);
        args.push_back("*");  // Auto-generate ID
        
        for (const auto& [key, value] : fields) {
            args.push_back(key);
            args.push_back(value);
        }
        
        // Convert to argv format
        for (const auto& arg : args) {
            argv.push_back(arg.c_str());
        }
        
        redisReply* reply = (redisReply*)redisCommandArgv(context_, argv.size(), argv.data(), nullptr);
        
        std::string message_id;
        if (reply && reply->type == REDIS_REPLY_STRING) {
            message_id = std::string(reply->str, reply->len);
        }
        
        if (reply) freeReplyObject(reply);
        return message_id;
    }
    
    // Read messages from stream
    std::vector<StreamMessage> xread(const std::string& stream_key, const std::string& start_id = "0", 
                                   int count = 10, int block_ms = 0) {
        if (!ensure_connected()) return {};
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        redisReply* reply;
        if (block_ms > 0) {
            // Blocking read: XREAD BLOCK timeout COUNT count STREAMS stream_key start_id
            reply = (redisReply*)redisCommand(context_, "XREAD BLOCK %d COUNT %d STREAMS %s %s",
                                            block_ms, count, stream_key.c_str(), start_id.c_str());
        } else {
            // Non-blocking read: XREAD COUNT count STREAMS stream_key start_id
            reply = (redisReply*)redisCommand(context_, "XREAD COUNT %d STREAMS %s %s",
                                            count, stream_key.c_str(), start_id.c_str());
        }
        
        std::vector<StreamMessage> messages = parse_xread_reply(reply);
        if (reply) freeReplyObject(reply);
        
        return messages;
    }
    
    // Read messages in time range
    std::vector<StreamMessage> xrange(const std::string& stream_key, const std::string& start = "-", 
                                    const std::string& end = "+", int count = -1) {
        if (!ensure_connected()) return {};
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        redisReply* reply;
        if (count > 0) {
            reply = (redisReply*)redisCommand(context_, "XRANGE %s %s %s COUNT %d",
                                            stream_key.c_str(), start.c_str(), end.c_str(), count);
        } else {
            reply = (redisReply*)redisCommand(context_, "XRANGE %s %s %s",
                                            stream_key.c_str(), start.c_str(), end.c_str());
        }
        
        std::vector<StreamMessage> messages = parse_xrange_reply(reply);
        if (reply) freeReplyObject(reply);
        
        return messages;
    }
    
    // Create consumer group
    bool xgroup_create(const std::string& stream_key, const std::string& group_name, 
                      const std::string& start_id = "$") {
        if (!ensure_connected()) return false;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        redisReply* reply = (redisReply*)redisCommand(context_, "XGROUP CREATE %s %s %s MKSTREAM",
                                                    stream_key.c_str(), group_name.c_str(), start_id.c_str());
        
        bool success = (reply && (reply->type == REDIS_REPLY_STATUS || reply->type == REDIS_REPLY_ERROR));
        if (reply && reply->type == REDIS_REPLY_ERROR) {
            std::string error(reply->str);
            success = (error.find("BUSYGROUP") != std::string::npos);  // Group already exists
        }
        
        if (reply) freeReplyObject(reply);
        return success;
    }
    
    // Read from consumer group
    std::vector<StreamMessage> xreadgroup(const std::string& group_name, const std::string& consumer_name,
                                        const std::string& stream_key, int count = 1, int block_ms = 0) {
        if (!ensure_connected()) return {};
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        redisReply* reply;
        if (block_ms > 0) {
            reply = (redisReply*)redisCommand(context_, "XREADGROUP GROUP %s %s BLOCK %d COUNT %d STREAMS %s >",
                                            group_name.c_str(), consumer_name.c_str(), block_ms, count, stream_key.c_str());
        } else {
            reply = (redisReply*)redisCommand(context_, "XREADGROUP GROUP %s %s COUNT %d STREAMS %s >",
                                            group_name.c_str(), consumer_name.c_str(), count, stream_key.c_str());
        }
        
        std::vector<StreamMessage> messages = parse_xread_reply(reply);
        if (reply) freeReplyObject(reply);
        
        return messages;
    }
    
    // Acknowledge message processing
    bool xack(const std::string& stream_key, const std::string& group_name, const std::string& message_id) {
        if (!ensure_connected()) return false;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        redisReply* reply = (redisReply*)redisCommand(context_, "XACK %s %s %s",
                                                    stream_key.c_str(), group_name.c_str(), message_id.c_str());
        
        bool success = (reply && reply->type == REDIS_REPLY_INTEGER && reply->integer > 0);
        if (reply) freeReplyObject(reply);
        
        return success;
    }
    
    // Get stream info
    std::map<std::string, std::string> xinfo_stream(const std::string& stream_key) {
        if (!ensure_connected()) return {};
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        redisReply* reply = (redisReply*)redisCommand(context_, "XINFO STREAM %s", stream_key.c_str());
        
        std::map<std::string, std::string> info;
        if (reply && reply->type == REDIS_REPLY_ARRAY) {
            for (size_t i = 0; i < reply->elements; i += 2) {
                if (i + 1 < reply->elements) {
                    std::string key(reply->element[i]->str, reply->element[i]->len);
                    std::string value(reply->element[i + 1]->str, reply->element[i + 1]->len);
                    info[key] = value;
                }
            }
        }
        
        if (reply) freeReplyObject(reply);
        return info;
    }

    // ✅ Existing operations (set, get, publish, etc.) remain the same...
    bool set(const std::string& key, const std::string& value, int ttl_seconds = 0) {
        if (!ensure_connected()) return false;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        redisReply* reply;
        if (ttl_seconds > 0) {
            reply = (redisReply*)redisCommand(context_, "SETEX %s %d %s", 
                                            key.c_str(), ttl_seconds, value.c_str());
        } else {
            reply = (redisReply*)redisCommand(context_, "SET %s %s", 
                                            key.c_str(), value.c_str());
        }
        
        bool success = (reply != nullptr && reply->type == REDIS_REPLY_STATUS);
        if (reply) freeReplyObject(reply);
        
        return success;
    }
    
    bool publish(const std::string& channel, const std::string& message) {
        if (!ensure_connected()) return false;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        redisReply* reply = (redisReply*)redisCommand(context_, "PUBLISH %s %s", 
                                                    channel.c_str(), message.c_str());
        
        bool success = (reply != nullptr && reply->type == REDIS_REPLY_INTEGER);
        if (reply) freeReplyObject(reply);
        
        return success;
    }

private:
    std::string host_;
    int port_;
    std::string password_;
    redisContext* context_;
    bool connected_;
    mutable std::mutex mutex_;
    
    bool ensure_connected() {
        if (!is_connected()) {
            return connect();
        }
        return true;
    }
    
    // Parse XREAD/XREADGROUP reply
    std::vector<StreamMessage> parse_xread_reply(redisReply* reply) {
        std::vector<StreamMessage> messages;
        
        if (!reply || reply->type != REDIS_REPLY_ARRAY) return messages;
        
        // XREAD returns: [[stream_name, [[id, [field, value, ...]], ...]]]
        for (size_t i = 0; i < reply->elements; i++) {
            redisReply* stream_reply = reply->element[i];
            if (stream_reply->type == REDIS_REPLY_ARRAY && stream_reply->elements >= 2) {
                redisReply* messages_reply = stream_reply->element[1];
                if (messages_reply->type == REDIS_REPLY_ARRAY) {
                    for (size_t j = 0; j < messages_reply->elements; j++) {
                        StreamMessage msg = parse_stream_message(messages_reply->element[j]);
                        if (!msg.id.empty()) {
                            messages.push_back(msg);
                        }
                    }
                }
            }
        }
        
        return messages;
    }
    
    // Parse XRANGE reply  
    std::vector<StreamMessage> parse_xrange_reply(redisReply* reply) {
        std::vector<StreamMessage> messages;
        
        if (!reply || reply->type != REDIS_REPLY_ARRAY) return messages;
        
        // XRANGE returns: [[id, [field, value, ...]], ...]
        for (size_t i = 0; i < reply->elements; i++) {
            StreamMessage msg = parse_stream_message(reply->element[i]);
            if (!msg.id.empty()) {
                messages.push_back(msg);
            }
        }
        
        return messages;
    }
    
    // Parse individual stream message: [id, [field, value, ...]]
    StreamMessage parse_stream_message(redisReply* msg_reply) {
        StreamMessage message;
        
        if (!msg_reply || msg_reply->type != REDIS_REPLY_ARRAY || msg_reply->elements < 2) {
            return message;
        }
        
        // Get message ID
        if (msg_reply->element[0]->type == REDIS_REPLY_STRING) {
            message.id = std::string(msg_reply->element[0]->str, msg_reply->element[0]->len);
            
            // Parse timestamp from ID
            size_t dash_pos = message.id.find('-');
            if (dash_pos != std::string::npos) {
                message.timestamp = std::stoull(message.id.substr(0, dash_pos));
            }
        }
        
        // Get fields
        redisReply* fields_reply = msg_reply->element[1];
        if (fields_reply->type == REDIS_REPLY_ARRAY) {
            for (size_t i = 0; i < fields_reply->elements; i += 2) {
                if (i + 1 < fields_reply->elements) {
                    std::string key(fields_reply->element[i]->str, fields_reply->element[i]->len);
                    std::string value(fields_reply->element[i + 1]->str, fields_reply->element[i + 1]->len);
                    message.fields[key] = value;
                }
            }
        }
        
        return message;
    }
};

// ✅ NEW: VLM Redis Stream Manager
class VLMRedisStreamManager {
public:
    VLMRedisStreamManager(const std::string& redis_host = "localhost", int redis_port = 6379)
        : redis_client_(redis_host, redis_port),
          vlm_stream_("vlm:results:stream"),
          frame_stream_("vlm:frames:stream"),
          consumer_group_("vlm_processors"),
          consumer_name_("deepstream_vlm") {
        
        if (!redis_client_.connect()) {
            std::cerr << "❌ Failed to connect to Redis for VLM streams" << std::endl;
            return;
        }
        
        // Create consumer groups
        redis_client_.xgroup_create(vlm_stream_, consumer_group_, "0");
        redis_client_.xgroup_create(frame_stream_, consumer_group_, "0");
        
        std::cout << "✅ VLM Redis Streams initialized" << std::endl;
    }
    
    // Add VLM result to stream
    std::string add_vlm_result(uint32_t frame_number, uint32_t source_id, 
                              const std::string& vlm_response, const std::string& model_name = "default") {
        std::map<std::string, std::string> fields = {
            {"frame_number", std::to_string(frame_number)},
            {"source_id", std::to_string(source_id)},
            {"vlm_response", vlm_response},
            {"model_name", model_name},
            {"timestamp", std::to_string(get_current_timestamp())},
            {"type", "vlm_result"}
        };
        
        return redis_client_.xadd(vlm_stream_, fields);
    }
    
    // Add frame metadata to stream
    std::string add_frame_metadata(uint32_t frame_number, uint32_t source_id, 
                                  uint32_t width, uint32_t height, const std::string& format = "NV12") {
        std::map<std::string, std::string> fields = {
            {"frame_number", std::to_string(frame_number)},
            {"source_id", std::to_string(source_id)},
            {"width", std::to_string(width)},
            {"height", std::to_string(height)},
            {"format", format},
            {"timestamp", std::to_string(get_current_timestamp())},
            {"type", "frame_metadata"}
        };
        
        return redis_client_.xadd(frame_stream_, fields);
    }
    
    // Read latest VLM results
    std::vector<StreamMessage> get_latest_vlm_results(int count = 10, int block_ms = 1000) {
        return redis_client_.xreadgroup(consumer_group_, consumer_name_, vlm_stream_, count, block_ms);
    }
    
    // Read VLM results in time range
    std::vector<StreamMessage> get_vlm_results_range(uint64_t start_timestamp, uint64_t end_timestamp, int count = 100) {
        std::string start_id = std::to_string(start_timestamp) + "-0";
        std::string end_id = std::to_string(end_timestamp) + "-0";
        return redis_client_.xrange(vlm_stream_, start_id, end_id, count);
    }
    
    // Get VLM results for specific source
    std::vector<StreamMessage> get_vlm_results_by_source(uint32_t source_id, int count = 50) {
        auto messages = redis_client_.xrange(vlm_stream_, "-", "+", count * 2);  // Get more to filter
        
        std::vector<StreamMessage> filtered;
        for (const auto& msg : messages) {
            if (msg.get_field_as<uint32_t>("source_id") == source_id) {
                filtered.push_back(msg);
                if (filtered.size() >= count) break;
            }
        }
        
        return filtered;
    }
    
    // Acknowledge processed message
    bool ack_message(const std::string& stream, const std::string& message_id) {
        return redis_client_.xack(stream, consumer_group_, message_id);
    }
    
    // Get stream statistics
    std::map<std::string, std::string> get_vlm_stream_stats() {
        return redis_client_.xinfo_stream(vlm_stream_);
    }
    
    std::map<std::string, std::string> get_frame_stream_stats() {
        return redis_client_.xinfo_stream(frame_stream_);
    }
    
    // Set custom stream names
    void configure_streams(const std::string& vlm_stream, const std::string& frame_stream, 
                          const std::string& consumer_group = "vlm_processors") {
        vlm_stream_ = vlm_stream;
        frame_stream_ = frame_stream;
        consumer_group_ = consumer_group;
        
        // Create new consumer groups
        redis_client_.xgroup_create(vlm_stream_, consumer_group_, "0");
        redis_client_.xgroup_create(frame_stream_, consumer_group_, "0");
    }
    
    bool is_connected() const {
        return redis_client_.is_connected();
    }

private:
    RedisClient redis_client_;
    std::string vlm_stream_;
    std::string frame_stream_;
    std::string consumer_group_;
    std::string consumer_name_;
    
    uint64_t get_current_timestamp() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
};

#endif // REDIS_CLIENT_H

/*
REDIS STREAMS BENEFITS FOR VLM:

✅ PERSISTENCE: Messages are stored, not lost like pub/sub
✅ ORDERING: Guaranteed message order with timestamps  
✅ CONSUMER GROUPS: Multiple workers can process streams
✅ ACKNOWLEDGMENT: Reliable message processing
✅ TIME QUERIES: Get results by time range
✅ SCALABILITY: Horizontal scaling with consumer groups
✅ BACKPRESSURE: Built-in flow control

USAGE EXAMPLES:

1. Basic VLM streaming:
   VLMRedisStreamManager vlm_stream;
   std::string msg_id = vlm_stream.add_vlm_result(123, 0, "Car on highway");
   
2. Process VLM results:
   auto results = vlm_stream.get_latest_vlm_results(10, 5000);  // 10 results, 5s timeout
   for (const auto& msg : results) {
       uint32_t frame = msg.get_field_as<uint32_t>("frame_number");
       std::string response = msg.get_field("vlm_response");
       vlm_stream.ack_message("vlm:results:stream", msg.id);
   }

3. Historical queries:
   uint64_t hour_ago = get_timestamp() - 3600000;
   auto historical = vlm_stream.get_vlm_results_range(hour_ago, get_timestamp());

REDIS CLI COMMANDS:

# Monitor stream in real-time
XREAD BLOCK 0 STREAMS vlm:results:stream $

# Get latest 10 messages
XRANGE vlm:results:stream - + COUNT 10

# Stream info
XINFO STREAM vlm:results:stream

# Consumer group info  
XINFO GROUPS vlm:results:stream
*/