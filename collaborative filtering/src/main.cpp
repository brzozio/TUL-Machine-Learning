#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>

#include <array>
#include <vector>
#include <unordered_map>

#include <random>

#include <future>

#define NUM_OF_MOVIES 200
#define NUM_OF_FEATURES 10
#define TRAINING_LOOPS 2000
#define LEARNING_RATE 0.001


enum error_code{
    OK = 0,
    unexpectedRepositoryStructure = 1,
    couldNotReadDataFile = 2,
    couldNotWriteDataFile = 3
};

std::pair<std::string, error_code> getRepositoryPath(){    

    std::string repo_path = std::filesystem::current_path().generic_string();
    std::size_t found_at = repo_path.find("/src");

    if (found_at!=std::string::npos) return std::pair(std::string(repo_path, 0, found_at+1), error_code::OK);

    return std::pair("",error_code::unexpectedRepositoryStructure);
}

std::pair<std::unordered_map<int, std::unordered_map<int, float>>, error_code> loadUserRatingsTrainData(const std::string &REPO_PATH){ 

    std::unordered_map<int, std::unordered_map<int, float>> user_movie_rating;

    std::ifstream file_user_ratings_train(REPO_PATH + "csv/train.csv");
    if (!file_user_ratings_train.is_open()) return std::pair(user_movie_rating, error_code::couldNotReadDataFile);
    

    std::string line;    
    int loaded_user_id = 0;
    int loaded_movie_id = 0;
    int loaded_rating = 0;
    std::string loaded_string;

    while(std::getline(file_user_ratings_train,line)){

        std::stringstream string_stream(line);

        // load entry ID and throw away
        std::getline(string_stream, loaded_string, ';');

        std::getline(string_stream, loaded_string, ';');
        loaded_user_id = std::stoi(loaded_string);
        
        std::getline(string_stream, loaded_string, ';');
        loaded_movie_id = std::stoi(loaded_string);

        std::getline(string_stream, loaded_string, ';');
        loaded_rating = std::stof(loaded_string);

        // std::unoredered_set::operator[]() adds default value if key does not exist so no checks needed
        user_movie_rating[loaded_user_id][loaded_movie_id] = loaded_rating;

    }

    return std::pair(user_movie_rating, error_code::OK);
}

std::pair<std::vector<std::vector<int>>, error_code> loadTask(const std::string &REPO_PATH){

    std::vector<std::vector<int>> task_data;

    std::ifstream file_user_ratings_train(REPO_PATH + "csv/task.csv");
    if (!file_user_ratings_train.is_open()) return std::pair(task_data, error_code::couldNotReadDataFile);
    
    std::string loaded_line;
    std::string loaded_string;

    while(std::getline(file_user_ratings_train,loaded_line)){
        
        std::vector<int> loaded_values;
        std::stringstream string_stream(loaded_line); 

        //loading integers up to NaN at the end of each line
        std::getline(string_stream, loaded_string, ';');
        loaded_values.push_back(std::stoi(loaded_string));
        std::getline(string_stream, loaded_string, ';');
        loaded_values.push_back(std::stoi(loaded_string));
        std::getline(string_stream, loaded_string, ';');
        loaded_values.push_back(std::stoi(loaded_string));

        task_data.push_back(loaded_values);
    }
    
    return std::pair(task_data, error_code::OK);
}

void fillRandom0to1(std::unordered_map<int, std::vector<float>>& output){

    std::uniform_real_distribution<> dis(0.0, 1.0);

    for(auto& [key, array]: output){
        for(auto& x: array){
            std::random_device rd;
            std::mt19937 gen(rd());
            x = dis(gen);
        }
    }
}

float predict(
    const std::vector<float>& paramId_param,
    const std::vector<float>& featId_feature
){
    float sum = 0.0;
    for(int i = 0; i < NUM_OF_FEATURES; i++){
        sum+=paramId_param[i]*featId_feature[i];
    }
    sum+=paramId_param[NUM_OF_FEATURES];
    if(sum > 5.0) return 5.0;
    if(sum < 0.0) return 0.0;
    return sum;
}

void tuneParamsForUser(const std::vector<int>& users,
    std::unordered_map<int, std::vector<float>>& user_paramId_param,
    const std::unordered_map<int, std::unordered_map<int, float>>& user_movie_rating,
    const std::unordered_map<int, std::vector<float>>& movie_featId_feature
){

    float derivative = 0;

    for(auto& user: users){

        std::array<float, NUM_OF_FEATURES+1> derivs{0};

        for(auto& [movie, rating]: user_movie_rating.at(user)){
            
            if(rating != -1){

                derivative = rating;
                derivative -= predict(user_paramId_param.at(user), movie_featId_feature.at(movie));

                for(int pid = 0; pid < NUM_OF_FEATURES; pid++){
                    derivs[pid] += derivative * movie_featId_feature.at(movie)[pid];
                }
                derivs[NUM_OF_FEATURES] += derivative;
            }
        }

        for(int pid = 0; pid < NUM_OF_FEATURES+1; pid++){
            user_paramId_param.at(user)[pid] += LEARNING_RATE*derivs[pid];
        }
    }
}

void tuneFeatsForMovie(const std::vector<int>& movies,
    std::unordered_map<int, std::vector<float>>& movie_featId_feature,
    const std::unordered_map<int, std::unordered_map<int, float>>& user_movie_rating,
    const std::unordered_map<int, std::vector<float>>& user_paramId_param
){

    float derivative = 0;

    for(auto& movie: movies){

        std::array<float, NUM_OF_FEATURES> derivs{0};

        for(auto& [user, movie_rating]: user_movie_rating){
            
            if(movie_rating.find(movie) != movie_rating.end()){

                derivative = movie_rating.at(movie);
                derivative -= predict(user_paramId_param.at(user), movie_featId_feature.at(movie));

                for(int fid = 0; fid < NUM_OF_FEATURES; fid++){
                    derivs[fid] += derivative * user_paramId_param.at(user)[fid];
                }
            }
        }

        for(int fid = 0; fid < NUM_OF_FEATURES; fid++){
            movie_featId_feature.at(movie)[fid] += LEARNING_RATE*derivs[fid];
        }
    }
}


const unsigned int NUM_OF_THREADS = std::thread::hardware_concurrency();

std::condition_variable cv_scheduler;
std::condition_variable cv_workers;
std::mutex mx;

unsigned int workersDone = 0;
std::vector<char> workPermit(NUM_OF_THREADS, 1);


void worker(
    bool *const escape, bool *const task, char *const permit, 
    const std::vector<int>& users, const std::vector<int>& movies,
    std::unordered_map<int, std::vector<float>>& user_paramId_param,
    std::unordered_map<int, std::vector<float>>& movie_featId_feature,
    const std::unordered_map<int, std::unordered_map<int, float>>& user_movie_rating
){
    while(true){

        {
            std::unique_lock<std::mutex> lk(mx);
            cv_workers.wait(lk, [permit]{ return *permit; });
            *permit = 0;
        }

        if(!*escape) return;

        if(*task) tuneParamsForUser(users, user_paramId_param, user_movie_rating, movie_featId_feature);        

        else tuneFeatsForMovie(movies, movie_featId_feature, user_movie_rating, user_paramId_param);

        {
            std::lock_guard<std::mutex> lk(mx);
            workersDone++;
        }
        cv_scheduler.notify_all();
    }
}
 
void scheduler(bool *const escape, bool *const task)
{
    int loopid = 0;
    while(*escape){
        {
            std::unique_lock<std::mutex> lk(mx);
            cv_scheduler.wait(lk, []{ return workersDone == NUM_OF_THREADS;}); 
            for(auto& permit: workPermit){
                permit = 1;
            }
            workersDone = 0;
            *task = (*task) ? false : true;
            loopid++;
            if(loopid >= TRAINING_LOOPS) *escape = false;
        }
        cv_workers.notify_all();
    }
    return;
}

void train(
    std::unordered_map<int, std::vector<float>>& user_paramId_param,
    std::unordered_map<int, std::vector<float>>& movie_featId_feature,
    const std::unordered_map<int, std::unordered_map<int, float>>& user_movie_rating
){

    bool training_mode = true;
    bool continue_training = true;

    std::vector<std::vector<int>> uid_split(NUM_OF_THREADS, std::vector<int>());
    int offset = 0;
    for(auto& [user, movie_rating]: user_movie_rating){
        uid_split[offset].push_back(user);
        offset++;
        if(offset >= NUM_OF_THREADS) offset = 0;
    }

    offset = 0;
    std::vector<std::vector<int>> mid_split(NUM_OF_THREADS, std::vector<int>());
    for(int movie = 1; movie <= 200; movie++){
        mid_split[offset].push_back(movie);
        offset++;
        if(offset >= NUM_OF_THREADS) offset = 0;
    }

    std::vector<std::thread> workers;

    for(int tid = 0; tid < NUM_OF_THREADS; tid++){
        workers.emplace_back(worker, &continue_training, &training_mode, &workPermit[tid],
        std::cref(uid_split[tid]), std::cref(mid_split[tid]),
        std::ref(user_paramId_param), std::ref(movie_featId_feature), 
        std::cref(user_movie_rating));
    }

    scheduler(&continue_training, &training_mode);

    for(auto& thread: workers){
        thread.join();
    }
    
    return;
}

error_code generateTask(
    const std::string &REPO_PATH,
    const std::unordered_map<int, std::vector<float>>& user_paramId_param,
    const std::unordered_map<int, std::vector<float>>& movie_featId_feature
){

    error_code ERROR_CODE = error_code::OK;
    std::vector<std::vector<int>> USER_MOVIE_TASK;

    std::tie(USER_MOVIE_TASK, ERROR_CODE) = loadTask(REPO_PATH);
    if(ERROR_CODE){
        std::cerr<<"USER RATINGS TASK DATASET NOT FOUND";
        return ERROR_CODE;
    }

    std::ofstream output_stream(REPO_PATH + "csv/submission.csv");
    if (!output_stream.is_open()){
        std::cerr<<"FAILED TO OPEN OUTPUT FILE";
        return error_code::couldNotWriteDataFile;
    }

    for(auto &entry: USER_MOVIE_TASK){

        for(auto &element: entry){
            output_stream<<element<<";";
        }

        output_stream<<std::roundf(predict(user_paramId_param.at(entry[1]), movie_featId_feature.at(entry[2])))<<std::endl;

    }

    return error_code::OK;
}

int main(int argc, char** argv){

    error_code ERROR_CODE = error_code::OK;

    std::string REPO_PATH = "";
    std::tie(REPO_PATH, ERROR_CODE) = getRepositoryPath();
    if(ERROR_CODE) {
        std::cerr<<ERROR_CODE;
        return ERROR_CODE;
    }

    std::unordered_map<int, std::unordered_map<int, float>> USER_MOVIE_RATING;
    std::tie(USER_MOVIE_RATING, ERROR_CODE) = loadUserRatingsTrainData(REPO_PATH);
    if(ERROR_CODE){
        std::cerr<<ERROR_CODE;
        return ERROR_CODE;
    }

    std::unordered_map<int, std::vector<float>> user_paramId_param;
    for(auto& [user,movie_rating]: USER_MOVIE_RATING){
        user_paramId_param[user] = std::vector<float>(NUM_OF_FEATURES+1, 0);
    }
    fillRandom0to1(user_paramId_param);

    std::unordered_map<int, std::vector<float>> movie_featId_feature;
    for(int movie = 1; movie <= NUM_OF_MOVIES; movie++){
        movie_featId_feature[movie] = std::vector<float>(NUM_OF_FEATURES, 0);
    }
    fillRandom0to1(movie_featId_feature);

    train(user_paramId_param, movie_featId_feature, USER_MOVIE_RATING);
    
    ERROR_CODE = generateTask(REPO_PATH,user_paramId_param,movie_featId_feature);
    if(ERROR_CODE) {
        std::cerr<<ERROR_CODE;
        return ERROR_CODE;
    }

    return error_code::OK;
}