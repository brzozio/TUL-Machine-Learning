// https://en.cppreference.com/w/cpp/thread/condition_variable/notify_all

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <thread>
#include <vector>

const auto NUM_OF_THREADS = std::thread::hardware_concurrency();

std::condition_variable cv_scheduler;
std::condition_variable cv_workers;

std::mutex mx;

unsigned int workersDone = 0;
std::vector<int> workPermit(NUM_OF_THREADS, 1);


void worker(int *const permit)
{
    while(true){
        {
            std::unique_lock<std::mutex> lock(mx);
            cv_workers.wait(lock, [permit]{ return *permit; });
            *permit = 0;
        }

        std::cerr << "begin work\n";

        std::this_thread::sleep_for(std::chrono::seconds(1));

        {
            std::lock_guard<std::mutex> lk(mx);
            workersDone++;
            std::cerr<<workersDone<<" done\t";
        }
        cv_scheduler.notify_all();
    }
}
 
void scheduler()
{
    while(true){
        {
            std::unique_lock<std::mutex> lock(mx);
            cv_scheduler.wait(lock, []{ return workersDone == NUM_OF_THREADS;}); 
            for(auto& permit: workPermit){
                permit = 1;
            }
            workersDone = 0;
        }
        cv_workers.notify_all();
    }
}
 
int main()
{
    std::vector<std::thread> threads;
    for(int id = 0; id < NUM_OF_THREADS; id++){
        threads.emplace_back(worker, &workPermit[id]);
    }
    scheduler();

}