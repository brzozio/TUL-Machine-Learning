#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include <future>

enum error_code{
    OK = 0,
    unexpectedRepositoryStructure = 1,
    couldNotReadDataFile = 2,
    couldNotWriteDataFile = 3
};

std::pair<std::string, error_code> get_repository_path(){    

    std::string repo_path = std::filesystem::current_path().generic_string();
    std::size_t found_at = repo_path.find("/src");

    if (found_at!=std::string::npos) return std::pair(std::string(repo_path, 0, found_at+1), error_code::OK);

    return std::pair("",error_code::unexpectedRepositoryStructure);
}

std::pair<std::unordered_map<int, std::unordered_map<int, int>>, error_code> load_user_ratings_train_data(const std::string &REPO_PATH){ 

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

std::pair<std::vector<std::vector<int>>, error_code> load_task(const std::string &REPO_PATH){

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
                static_cast<float>(sum_rating_difference) * (1.0 - static_cast<float>(hit_count) / 
                static_cast<float>(primaryUser.second.size() + secondaryUser.second.size() - hit_count));

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

int predict_rating(const int &user_id, const int &movie_id, const int &match_count,
const std::unordered_map<int, std::vector<int>> &user_closestUser_user, 
const std::unordered_map<int, std::unordered_map<int, int>> &user_movie_rating){
    
        int closest_match = -1;
        std::vector<int> match_ratings;
        match_ratings.reserve(match_count);

        while((match_ratings.size() < match_count) and (closest_match < static_cast<int>(user_closestUser_user.at(user_id).size()))){
            
            closest_match++;
            auto iskey = user_movie_rating.at( user_closestUser_user.at(user_id)[closest_match] ).find(movie_id);

            while(iskey == user_movie_rating.at( user_closestUser_user.at(user_id)[closest_match] ).end()){
                
                closest_match++;
                iskey = user_movie_rating.at( user_closestUser_user.at(user_id)[closest_match] ).find(movie_id);

            }

            match_ratings.push_back(user_movie_rating.at( user_closestUser_user.at(user_id)[closest_match] ).at(movie_id));

        }

        int possible_ratings[6] = {0, 0, 0, 0, 0, 0};

        for(auto &match: match_ratings){
            possible_ratings[match]+=2;
        }

        int max_rating = 0;
        for(int rating = 0; rating < 6; rating++){
            if(possible_ratings[max_rating] < possible_ratings[rating]){
                max_rating = rating;
            }
        }
        for(int rating = 0; rating < 6; rating++){
            if (possible_ratings[max_rating] == possible_ratings[rating]){
                for(auto &match_rating: match_ratings){
                    if(match_rating == max_rating){
                        break;
                    }
                    else if(match_rating == rating){
                        max_rating = rating;
                        break;
                    }
                }
            }
        }

        return max_rating;
}

error_code generate_task(const std::string &REPO_PATH,
const std::unordered_map<int, std::vector<int>> &user_closestUser_user, const std::unordered_map<int, std::unordered_map<int, int>> &user_movie_rating){

    error_code ERROR_CODE = error_code::OK;
    std::vector<std::vector<int>> USER_MOVIE_TASK;

    std::tie(USER_MOVIE_TASK, ERROR_CODE) = load_task(REPO_PATH);
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

        output_stream<<predict_rating(entry[1], entry[2], 3, user_closestUser_user, user_movie_rating)<<std::endl;

    }

    return error_code::OK;
}

int main(int argc, char** argv){

    error_code ERROR_CODE = error_code::OK;

    std::string REPO_PATH = "";
    std::tie(REPO_PATH, ERROR_CODE) = get_repository_path();
    if(ERROR_CODE) {
        std::cerr<<ERROR_CODE;
        return ERROR_CODE;
    }

    // hashmap storage for unified internal and external ID system
    std::unordered_map<int, std::unordered_map<int, int>> USER_MOVIE_RATING;
    std::tie(USER_MOVIE_RATING, ERROR_CODE) = load_user_ratings_train_data(REPO_PATH);
    if(ERROR_CODE){
        std::cerr<<ERROR_CODE;
        return ERROR_CODE;
    }

    auto user_user_distance = get_user_distances(USER_MOVIE_RATING);

    auto user_closestUser_user = get_user_closest_users(user_user_distance);
    
    ERROR_CODE = generate_task(REPO_PATH, user_closestUser_user, USER_MOVIE_RATING);
    if(ERROR_CODE) {
        std::cerr<<ERROR_CODE;
        return ERROR_CODE;
    }

    return error_code::OK;
}