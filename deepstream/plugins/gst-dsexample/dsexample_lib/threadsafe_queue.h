
#ifndef THREAD_SAFE_QUEUE_H_
#define THREAD_SAFE_QUEUE_H_

#include <memory>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

template <typename T>
class ThreadSafeQueue {
 public:
  ThreadSafeQueue() = default;

  void push(T new_value) {
    auto data = std::make_shared<T>(std::move(new_value)); // this can throw, but it's OK
    std::lock_guard<std::mutex> lock(m_);
    data_queue_.push(data);
    cond_.notify_one();
  }

  void share_push(std::shared_ptr<T> data) {
    std::lock_guard<std::mutex> lock(m_);
    data_queue_.push(data);
    cond_.notify_one();
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(m_);
    return data_queue_.empty();
  }

  int size() const {
    std::lock_guard<std::mutex> lock(m_);
    return data_queue_.size();
  }

  void wait_and_pop(T &value) {
    std::unique_lock<std::mutex> lock(m_);
    cond_.wait(lock, [this] {
      return !data_queue_.empty();
    });

    value = std::move(*data_queue_.front());
    data_queue_.pop();
  }

  bool try_pop(T &value) {
    std::lock_guard<std::mutex> lock(m_);
    if (data_queue_.empty()) {
      return false;
    }

    value = std::move(*data_queue_.front());
    data_queue_.pop();
    return true;
  }

  std::shared_ptr<T> wait_and_pop() {
    std::unique_lock<std::mutex> lock(m_);
    cond_.wait(lock, [this] {
      return !data_queue_.empty() || is_terminated_;
    });

    if (is_terminated_) {
      return nullptr;
    }

    auto res = data_queue_.front(); // safe, cannot throw
    data_queue_.pop();
    return res;
  }

  std::shared_ptr<T> try_pop() {
    std::lock_guard<std::mutex> lock(m_);
    if (data_queue_.empty()) {
      return {};
    }

    auto res = data_queue_.front();
    data_queue_.pop();
    return res;
  }

  void terminate() {
    is_terminated_ = true;
    cond_.notify_all();
  }

 private:
  mutable std::mutex m_{};
  std::queue<std::shared_ptr<T>> data_queue_{};
  std::condition_variable cond_{};
  std::atomic<bool> is_terminated_{false};
};

#endif //THREAD_SAFE_QUEUE_H_
