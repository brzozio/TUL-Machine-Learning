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

    for(auto &primaryUser: user_movie_rating){
        for(auto &secondaryUser: user_movie_rating){

            int sum_rating_difference = 0;
            int hit_count = 0;

            for(auto &movie_user_1: primaryUser.second){

                auto movie_user_2 = secondaryUser.second.find(movie_user_1.first);

                if(!(movie_user_2 == secondaryUser.second.end())){
                    sum_rating_difference += abs(movie_user_1.second - movie_user_2->second);
                    hit_count++;
                }
            }

            if(hit_count != 0) user_user_distance[primaryUser.first][secondaryUser.first] = 
                float(sum_rating_difference) * (1.0 - float(hit_count) / float(primaryUser.second.size() + secondaryUser.second.size() - hit_count));

            else user_user_distance[primaryUser.first][secondaryUser.first] = -1.0;
        }
    }

    return user_user_distance;
}

std::unordered_map<int, std::vector<int>> get_user_closest_users(const std::unordered_map<int, std::unordered_map<int, float>> &user_user_distance){
    std::unordered_map<int, std::vector<int>> user_closetUser_user;

    for(auto &primaryUser: user_user_distance){

        std::vector<std::pair<int, float>> user_distance;

        for(auto &secondaryUser: primaryUser.second){
            user_distance.push_back(secondaryUser);
        }

        std::sort(user_distance.begin(), user_distance.end(), 
            [](const std::pair<int,float> &x1, const std::pair<int,float> &x2){
                return x1.second < x2.second;
            });

        int first_valid_uid = 0;
        while(user_distance[first_valid_uid].first != primaryUser.first){
            first_valid_uid++;
        }
        first_valid_uid++;

        for(int uid = first_valid_uid; uid < user_distance.size(); uid++){
            user_closetUser_user[primaryUser.first].push_back(user_distance[uid].first);
        }

    }

    return user_closetUser_user;
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
    std::unordered_map<int, std::unordered_map<int, int>> USER_MOVIE_RATING;
    std::tie(USER_MOVIE_RATING, ERROR_CODE) = load_user_ratings_train_data(REPO_PATH);

    if(ERROR_CODE){
        std::cerr<<"USER RATINGS TRAINING DATASET NOT FOUND";
        return ERROR_CODE;
    }

    auto user_user_distance = get_user_distances(USER_MOVIE_RATING);

    auto user_closestUser_user = get_user_closest_users(user_user_distance);

        
    std::ofstream output_stream(REPO_PATH + "csv/user_closestUser_user.csv");
    if (!output_stream.is_open()){
        std::cerr<<"FAILED TO OPEN OUTPUT FILE";
        return 3;
    }

    for(auto &element: user_closestUser_user){
        output_stream<<element.first<<";";
        for(auto &mid: element.second){
            output_stream<<mid<<";";
        }
        output_stream<<std::endl;
    }

    output_stream.close();

    return ERROR_CODE;
}