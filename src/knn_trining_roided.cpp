#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>

enum ErrorCode {
    UNSUPPORTED_REPO_STRUCTURE = 1,
    FILE_NOT_FOUND = 2
};

#define NUM_OF_FEATURES 5
#define NUM_OF_MOVIES 200

std::string get_repository_path(){    
    std::string repo_path = std::filesystem::current_path().generic_string();
    std::size_t found_at = repo_path.find("/src/");
    if (found_at!=std::string::npos) return std::string(repo_path, 0, found_at+1);
    return "";
}

int main(int argc, char** argv){

    const std::string REPO_PATH = get_repository_path();    
    if(REPO_PATH == "") return ErrorCode::UNSUPPORTED_REPO_STRUCTURE;

    std::ifstream file(REPO_PATH + "csv/MOVIE_DISTANCE_TREE.csv");
    if (!file.is_open()) return ErrorCode::FILE_NOT_FOUND;
    

    float MOVIE_DISTANCE_TREE[NUM_OF_MOVIES][NUM_OF_MOVIES][NUM_OF_FEATURES];

    std::string line;
    int movie_1_id = 0, movie_2_id = 0, feature_id = 0;
    while(std::getline(file,line)){
        
        MOVIE_DISTANCE_TREE[movie_1_id][movie_2_id][feature_id] = std::stof(line);

        feature_id++;
        if(feature_id >= NUM_OF_FEATURES){
            movie_2_id ++;
            feature_id = 0;
        }
        if(movie_2_id >= NUM_OF_MOVIES){
            movie_1_id ++;
            movie_2_id = 0;
        }
    }

    return 0;
}