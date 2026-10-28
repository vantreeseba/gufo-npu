#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>

// One generation at a time, the rest waiting in arrival order in a bounded
// queue.
namespace gufo_npu::serve {

class Scheduler {
public:
  explicit Scheduler(int max_queue) : max_queue_(max_queue) {}

  class Ticket {
  public:
    ~Ticket() { scheduler_.Release(slot_); }
    Ticket(const Ticket&) = delete;
    Ticket& operator=(const Ticket&) = delete;

    // Blocks until this request may run. Returns false, giving up its
    // place, if `abandoned` turns true while it waits (polled every 50 ms).
    bool Wait(const std::function<bool()>& abandoned) {
      return scheduler_.Wait(slot_, abandoned);
    }

  private:
    friend class Scheduler;
    struct Slot {
      bool granted = false;
      bool done = false;
    };
    Ticket(Scheduler& scheduler, std::shared_ptr<Slot> slot)
        : scheduler_(scheduler), slot_(std::move(slot)) {}

    Scheduler& scheduler_;
    std::shared_ptr<Slot> slot_;
  };

  // A place in the queue, or nullptr when max_queue requests already wait.
  std::unique_ptr<Ticket> Enter() {
    std::lock_guard lock(mutex_);
    if (static_cast<int>(waiting_.size()) >= max_queue_ &&
        (running_ || !waiting_.empty())) {
      return nullptr;
    }
    auto slot = std::make_shared<Ticket::Slot>();
    waiting_.push_back(slot);
    GrantLocked();
    return std::unique_ptr<Ticket>(new Ticket(*this, std::move(slot)));
  }

  int waiting() const {
    std::lock_guard lock(mutex_);
    return static_cast<int>(waiting_.size());
  }
  bool running() const {
    std::lock_guard lock(mutex_);
    return running_;
  }

private:
  using Slot = Ticket::Slot;

  void GrantLocked() {
    if (!running_ && !waiting_.empty()) {
      waiting_.front()->granted = true;
      waiting_.pop_front();
      running_ = true;
      changed_.notify_all();
    }
  }

  bool Wait(const std::shared_ptr<Slot>& slot,
            const std::function<bool()>& abandoned) {
    std::unique_lock lock(mutex_);
    while (!slot->granted) {
      changed_.wait_for(lock, std::chrono::milliseconds(50));
      if (!slot->granted && abandoned && abandoned()) {
        Leave(slot);
        return false;
      }
    }
    return true;
  }

  void Release(const std::shared_ptr<Slot>& slot) {
    std::lock_guard lock(mutex_);
    Leave(slot);
  }

  // Under the lock: frees the generation slot or the queue place.
  void Leave(const std::shared_ptr<Slot>& slot) {
    if (slot->done) {
      return;
    }
    slot->done = true;
    if (slot->granted) {
      running_ = false;
    } else {
      waiting_.erase(std::find(waiting_.begin(), waiting_.end(), slot));
    }
    GrantLocked();
  }

  const int max_queue_;
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<std::shared_ptr<Slot>> waiting_;
  bool running_ = false;
};

}  // namespace gufo_npu::serve
