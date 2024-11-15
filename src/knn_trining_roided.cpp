#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <algorithm>

#define NUM_OF_FEATURES 5
#define NUM_OF_MOVIES 200

std::string get_repository_path(){    
    std::string repo_path = std::filesystem::current_path().generic_string();
    std::size_t found_at = repo_path.find("/src");
    if (found_at!=std::string::npos) return std::string(repo_path, 0, found_at+1);
    return "";
}

// calculates the relative distances of all pairs of movies in 'distance_coefficients'
// as a linear combination of relative distance coefficients and the provided 'metric'
// stores the result in 'scaled_dist_tensor' and sorts a matrix of movie_ids according
// to their new relative distances storing the result in 'destination_tensor'
inline void argsort_scaled_distances(int (&destiantion_tensor)[NUM_OF_MOVIES][NUM_OF_MOVIES], float (&scaled_dist_tensor)[NUM_OF_MOVIES][NUM_OF_MOVIES],
    const std::vector<std::vector<std::vector<float>>> (&distance_coefficients), const float (&metric)[NUM_OF_FEATURES]){
        
    for(int movie_1_id = 0; movie_1_id < NUM_OF_MOVIES; movie_1_id++){
        for(int feature_id = 0; feature_id < NUM_OF_FEATURES; feature_id++){
            for(int movie_2_id = 0; movie_2_id < NUM_OF_MOVIES; movie_2_id++){
                scaled_dist_tensor[movie_1_id][movie_2_id] += metric[feature_id]*distance_coefficients[movie_1_id][feature_id][movie_2_id];
            }
        }
    }

    // setting self distances to -1 to guarantee the order of arg sort 
    // in edge case of two scaled distances being 0, for example two
    // movies with same director and metric = 1 for (director-director) and 0 for others
    for(int movie_1_id = 0; movie_1_id < NUM_OF_MOVIES; movie_1_id++){
        scaled_dist_tensor[movie_1_id][movie_1_id] = -1.0;     
        for(int movie_2_id = 0; movie_2_id < NUM_OF_MOVIES; movie_2_id++){
            destiantion_tensor[movie_1_id][movie_2_id] = movie_2_id+1;
        }
    }

    for(int movie_id = 0; movie_id < NUM_OF_MOVIES; movie_id++){
        std::stable_sort(destiantion_tensor[movie_id], destiantion_tensor[movie_id]+NUM_OF_MOVIES,
            [&scaled_dist_tensor, &movie_id](const size_t &neigh_1_id, const size_t &neigh_2_id){ 
                return scaled_dist_tensor[movie_id][neigh_1_id-1] < scaled_dist_tensor[movie_id][neigh_2_id-1];
            });
    }
}

int main(int argc, char** argv){

    const std::string REPO_PATH = get_repository_path();    
    if(REPO_PATH == "") {
        std::cout<<"UNSUPPORTED REPO STRUCTURE";
        return 1;
    }

    std::ifstream file(REPO_PATH + "csv/MOVIE_DISTANCE_TREE.csv");
    if (!file.is_open()){
        std::cout<<"FILE NOT FOUND";
        return 2;
    } 
    
    // indexes are swapped to capitalize on prefetching and
    // loop unrolling via parralelization of linear combinations
    std::vector<std::vector<std::vector<float>>> MOVIE_DISTANCE_COEFFICIENTS_TENSOR(
        NUM_OF_MOVIES, std::vector<std::vector<float>>(NUM_OF_FEATURES, std::vector<float>(NUM_OF_MOVIES, 0)));

    std::string line;
    {
        int movie_1_id = 0, movie_2_id = 0, feature_id = 0;
        while(std::getline(file,line)){
            
            MOVIE_DISTANCE_COEFFICIENTS_TENSOR[movie_1_id][feature_id][movie_2_id] = std::stof(line);

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
    } file.close();
    
    float movie_distance_tensor[NUM_OF_MOVIES][NUM_OF_MOVIES] = {0};
    int movie_nearest_neighbours[NUM_OF_MOVIES][NUM_OF_MOVIES] = {0};

    argsort_scaled_distances(movie_nearest_neighbours, movie_distance_tensor, MOVIE_DISTANCE_COEFFICIENTS_TENSOR, {0.2,0.2,0.2,0.2,0.2});

    // for(int i = 0; i < 200; i++){
    //     std::cout<<movie_nearest_neighbours[0][i]<<":\t"<<movie_distance_tensor[0][movie_nearest_neighbours[0][i]-1]<<"\n";
    // }

    // std::cout<<"\n------------\n";
    
    // for(int i = 0; i < 200; i++){
    //     std::cout<<movie_nearest_neighbours[1][i]<<":\t"<<movie_distance_tensor[1][movie_nearest_neighbours[1][i]-1]<<"\n";
    // }

    // std::cout<<"\n------------\n";

    // for(int i = 0; i < 200; i++){
    //     std::cout<<movie_nearest_neighbours[2][i]<<":\t"<<movie_distance_tensor[2][movie_nearest_neighbours[2][i]-1]<<"\n";
    // }
    
    // std::string temp;
    // std::getline(std::cin, temp);

    return 0;
}