#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include <future>

std::pair<std::string, int> get_repository_path(){    
    std::string repo_path = std::filesystem::current_path().generic_string();
    std::size_t found_at = repo_path.find("/src");
    if (found_at!=std::string::npos) return std::pair(std::string(repo_path, 0, found_at+1), 0);
    return std::pair("",1);
}

std::pair<std::unordered_map<int, std::unordered_map<int, int>>, int> load_user_ratings_train_data(const std::string &REPO_PATH){ 

    std::unordered_map<int, std::unordered_map<int, int>> user_movie_rating;

    std::ifstream file_user_ratings_train(REPO_PATH + "csv/train.csv");
    if (!file_user_ratings_train.is_open()) return std::pair(user_movie_rating, 2);
    

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

    return std::pair(user_movie_rating, 0);
}

// for each user computes the rating distance to all other users defined as a
// sum of absolute differences between ratings for all common movies
// multiplied by the number of movies unique to each user divided by 
// the number of movies unique to both users
std::unordered_map<int, std::unordered_map<int, float>> get_user_distances(const std::unordered_map<int, std::unordered_map<int, int>> &user_movie_rating){
    
    std::unordered_map<int, std::unordered_map<int, float>> user_user_distance;

    for(auto &primary_user: user_movie_rating){
        for(auto &secondary_user: user_movie_rating){

            int sum_rating_difference = 0;
            int hit_count = 0;

            for(auto &movie_user_1: primary_user.second){

                auto movie_user_2 = secondary_user.second.find(movie_user_1.first);

                if(!(movie_user_2 == secondary_user.second.end())){
                    sum_rating_difference += abs(movie_user_1.second - movie_user_2->second);
                    hit_count++;
                }
            }

            if(hit_count != 0) user_user_distance[primary_user.first][secondary_user.first] = 
                float(sum_rating_difference) * (1.0 - float(hit_count) / float(primary_user.second.size() + secondary_user.second.size() - hit_count));

            else user_user_distance[primary_user.first][secondary_user.first] = -1.0;
        }
    }

    return user_user_distance;
}

int main(int argc, char** argv){

    int ERROR_CODE = 0;

    std::string REPO_PATH = "";
    std::tie(REPO_PATH, ERROR_CODE) = get_repository_path();

    if(ERROR_CODE) {
        std::cerr<<"UNSUPPORTED REPO STRUCTURE";
        return ERROR_CODE;
    }

    // hashmap storage for unified internal and external ID system
    // L"user_id".L"movie_id"."rating"
    std::unordered_map<int, std::unordered_map<int, int>> USER_MOVIE_RATING;
    std::tie(USER_MOVIE_RATING, ERROR_CODE) = load_user_ratings_train_data(REPO_PATH);

    if(ERROR_CODE){
        std::cerr<<"USER RATINGS TRAINING DATASET NOT FOUND";
        return ERROR_CODE;
    }

    // L"primary_user_id".L"n-th_closest_user"."secondary_user_id"
    auto USER_CLOSEST = get_user_distances(USER_MOVIE_RATING);

    // for(auto &primary_hashmap: USER_CLOSEST){
    //     std::cout<<"\n\n"<<primary_hashmap.first<<":\n";
    //     for(auto &secondary_hashmap: primary_hashmap.second){
    //         std::cout<<"("<<secondary_hashmap.first<<":"<<secondary_hashmap.second<<"),  ";
    //     }
    // }
    
    // const int THREAD_COUNT = [](){
    //     auto temp = std::thread::hardware_concurrency();
    //     if(temp > 0) return temp;
    //     return 1u;
    // }();

    
    // std::array<std::string, NUM_OF_USERS> output_data{""}; 


    // std::ofstream output_stream(REPO_PATH + "csv/TRENNING_PARAMETERS.csv");
    // if (!output_stream.is_open()){
    //     std::cerr<<"FAILED TO OPEN OUTPUT FILE";
    //     return 4;
    // }

    // for(auto &s: output_data){
    //     output_stream<<s<<"\n";
    // }

    // output_stream.close();

    return ERROR_CODE;
}