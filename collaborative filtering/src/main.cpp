#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>

#include <array>
#include <vector>
#include <unordered_map>

#include <random>
#include <algorithm>

#include <future>

#define NUM_OF_MOVIES 200
#define NUM_OF_USERS 358
#define NUM_OF_FEATURES 10


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

std::pair<std::unordered_map<int, std::unordered_map<int, int>>, error_code> loadUserRatingsTrainData(const std::string &REPO_PATH){ 

    std::unordered_map<int, std::unordered_map<int, int>> user_movie_rating;

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
        loaded_rating = std::stoi(loaded_string);

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

std::array<std::array<float, NUM_OF_MOVIES>, NUM_OF_USERS> transformRatings(const std::unordered_map<int, std::unordered_map<int, int>>& user_movie_rating){
    std::array< std::array<float, NUM_OF_MOVIES>, NUM_OF_USERS> output;

    for(auto& y: output){
        for(auto& x:y){
            x = -1;
        }
    }

    int uid=0;
    for(auto& [user, movie_rating]: user_movie_rating){
        for(auto& [movie, rating]: movie_rating){
            output[uid][movie-1] = static_cast<float>(rating);
        }
        uid++;
    }
    return output;
}

std::array<std::array<float, NUM_OF_MOVIES>, NUM_OF_USERS> getRatingPresenceIndicator(const std::unordered_map<int, std::unordered_map<int, int>>& user_movie_rating){
    std::array< std::array<float, NUM_OF_MOVIES>, NUM_OF_USERS> output;

    for(auto& y: output){
        for(auto& x:y){
            x = 0;
        }
    }

    int uid=0;
    for(auto& [user, movie_rating]: user_movie_rating){
        for(auto& [movie, rating]: movie_rating){
            output[uid][movie-1] = 1;
        }
        uid++;
    }
    return output;
}

template<size_t x1, size_t x2>
void fillRandom0to1(std::array<std::array<float, x1>, x2>& output){

    std::uniform_real_distribution<> dis(0.0, 1.0);

    for(auto& y: output){
        for(auto& x: y){
            std::random_device rd;
            std::mt19937 gen(rd());
            x = dis(gen);
        }
    }
}


void tuneParamsForUser(
    std::array<float,NUM_OF_FEATURES+1>& paramId_param,
    const std::array<std::array<float, NUM_OF_MOVIES>, NUM_OF_USERS>& localUserId_localMovieId_indicator,
    const std::array<std::array<float, NUM_OF_MOVIES>, NUM_OF_USERS>& localUserId_localMovieId_rating,
    const std::array<std::array<float,NUM_OF_FEATURES>, NUM_OF_MOVIES>& movie_featId_feature
){

}

void tuneFeatsForMovie(
    std::array<float,NUM_OF_FEATURES>& featId_feature,
    const std::array<std::array<float, NUM_OF_MOVIES>, NUM_OF_USERS>& localUserId_localMovieId_indicator,
    const std::array<std::array<float, NUM_OF_MOVIES>, NUM_OF_USERS>& localUserId_localMovieId_rating,
    const std::array<std::array<float,NUM_OF_FEATURES+1>, NUM_OF_USERS>& user_paramId_param
){

}


int predict_rating(const int &user_id, const int &movie_id
){
        int max_rating = 0;
        return max_rating;
}

error_code generateTask(
    const std::string &REPO_PATH
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

        output_stream<<predict_rating(entry[1], entry[2])<<std::endl;

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

    std::unordered_map<int, std::unordered_map<int, int>> USER_MOVIE_RATING;
    std::tie(USER_MOVIE_RATING, ERROR_CODE) = loadUserRatingsTrainData(REPO_PATH);
    if(ERROR_CODE){
        std::cerr<<ERROR_CODE;
        return ERROR_CODE;
    }

    const auto localUserId_localMovieId_indicator = getRatingPresenceIndicator(USER_MOVIE_RATING);

    const auto localUserId_localMovieId_rating = transformRatings(USER_MOVIE_RATING);

    std::array<std::array<float,NUM_OF_FEATURES+1>, NUM_OF_USERS> user_paramId_param;
    fillRandom0to1(user_paramId_param);

    std::array<std::array<float,NUM_OF_FEATURES>, NUM_OF_MOVIES> movie_featId_feature;
    fillRandom0to1(movie_featId_feature);



    
    ERROR_CODE = generateTask(REPO_PATH);
    if(ERROR_CODE) {
        std::cerr<<ERROR_CODE;
        return ERROR_CODE;
    }

    return error_code::OK;
}