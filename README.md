# Billboard

## MSD
The Million Songs Dataset, but we only keep the Additional/subset_unique_tracks.txt file since we only need the title and artist name for each tracks.

If you don't care how we fetch the data, just ignore this folder.

## tmp
We store all the temperory file here. The order is:
* hot_100.xlsx: inlcudes all the tracks in hot-100 from 1990-01-01 to 2019-03-21
* MSD_10000.xlsx: includes all the tracks (10000 in total) in MSD dataset
* feature_1990_2019.xlsx: includes features we extract from spotify API for all the tracks in hot_100.xlsx and MSD_10000.xlsx
* feature_complete_1990_2019.xlsx: here we add "artist_score" feature to feature_1990_2019
* feature_complete_normalized_1990_2019: we normalized the features to have zero mean and one std in feature_complete_1990_2019