#include <string>
#include <vector>
#include <sstream> //istringstream
#include <iostream> // cout
#include <fstream> // ifstream
#include "TalassConfig.h"
#include "ClanHandle.h"
#include "SegmentationHandle.h"
#include "SimplificationHandle.h"
#include "FeatureSegmentation.h"
#include "FeatureHierarchy.h"
#include "FeatureData.h"
#include "Segment.h"
#include "FeatureElement.h"

const char *filename = "CondensedOutput.family";

using namespace std;
using namespace TopologyFileFormat;
/**
 * Reads csv file into table, exported as a vector of vector of doubles.
 * @param inputFileName input file name (full path).
 * @return data as vector of vector of doubles.
 */
vector< vector<double> > parse2DCsvFile(string inputFileName) {

    vector< vector<double> > data;
    ifstream inputFile(inputFileName);
    int l = 0;

    while (inputFile) {
        l++;
        string s;
        if (!getline(inputFile, s)) break;
        if (s[0] != '#') {
            istringstream ss(s);
            vector<double> record;

            while (ss) {
                string line;
                if (!getline(ss, line, ','))
                    break;
                try {
                    record.push_back(stof(line));
                }
                catch (const std::invalid_argument e) {
                    cout << "NaN found in file " << inputFileName << " line " << l
                         << endl;
                    e.what();
                }
            }

            data.push_back(record);
        }
    }

    if (!inputFile.eof()) {
        cerr << "Could not read file " << inputFileName << "\n";
        __throw_invalid_argument("File not found.");
    }

    return data;
}

int main()
{
    // Read csv File
    // Format:
    // [self, 'Parent', 'x_pos', 'y_pos', 'Born', 'Died']
    vector< vector<double> > clust_data = parse2DCsvFile("../utilities/condensed_tree_lambda_noisy.csv");

    // Total number of nodes
    int num_nodes = clust_data.size();
    int num_points = 0;
    // Node hierarchy
    vector<FeatureElement> features(num_nodes,FeatureElement(SINGLE_REPRESENTATIVE));

    for (int i=0; i<num_nodes; i++) {
      /*if (i==1500) {
        cout << "Looking at 1500.." << endl;
        for (auto x : clust_data[i])
          cout << x << " ";
        cout << endl;
      }*/
      cout << "i: " << i << ", born: " << clust_data[i][4] << ", died: " << clust_data[i][5] << ", parent: "<< clust_data[i][1];
      cout << endl;
      if (clust_data[i][2] != -1){
        num_points += 1;
      }
      features[i].lifeTime(clust_data[i][4], clust_data[i][5]);
      features[i].direction(true);
      if (clust_data[i][1] != -1){
        features[i].addLink( (int) clust_data[i][1]);
      }
    }

    cout << "Adding the topmost node.." << endl;
    for (auto x : clust_data[num_nodes-1])
      cout << x << " ";
    cout << endl;

    // Node without parent missing has only 5 values
    //features[num_nodes-1].lifeTime(clust_data[num_nodes-1][3], 0.9);
    //features[num_nodes-1].direction(true);

    // Now we save this to a file

    // The name of the family file
    ClanHandle clan(filename);

    // The name of the data set (all time steps must have
    // the same data set name)
    clan.dataset("TestDataSet");

    // Create the family which is the internal name for the collection
    // of all the information related to a time step
    FamilyHandle family;

    // The index of this time step
    family.timeIndex(0);

    // The actual floating point real time. Often this is identical
    // to the index but it does not have to be. Also this allows to
    // accept time series with unevenly distributed time steps
    family.time(1.0);

    // The name of the threshold parameter we are using
    family.variableName("ParameterName");

    // The range of the parameter
    family.range(0,375.);

    // Now we create the actual hierarchy in form of a simplification
    // sequence. As before we first create the handle
    SimplificationHandle simplification;

    // Set some more meta data
    simplification.metric("Threshold");
    simplification.fileType(SINGLE_REPRESENTATIVE);
    simplification.setRange(0,375.);
    simplification.encoding(true); // For testing purposes we write this in ascii

    // Now we assign the data to this handle. To comply with the API we need to
    // have a type derived from FileData
    Data<FeatureElement> data(&features); // This does not copy the data but is simply a pointer

    simplification.setData(&data); // Again this only passes pointers


    // Now we attach the hierarchy to the family
    family.add(simplification);

    // and the family to the clan
    clan.add(family);

    // And finally we write the clan to file
    clan.write();


    // Now let's figure out how to add some geometric points to act as elements of each cluster

    // First we create a 10x10 grid of points (standard row major layout)
    vector<FunctionType> coords(num_points*3);

    for (int i=0;i<num_points;i++) {
      coords[3*i] = clust_data[i][2]/10;     // x-coord
      coords[3*i + 1] = 0;  //clust_data[i][2]/10; // y-coord
      coords[3*i + 2] = clust_data[i][3]/10; // z-coord
      //cout << "i: " << i << ", x: " << clust_data[i][2]/10 << ", z: " << clust_data[i][3]/10;
      //cout << endl;

    }

    // And we add those to a geometry handle
    GeometryHandle geometry;
    geometry.dimension(3); // We could have done this in 2 dimensions
    geometry.encoding(true); // Make things ascii for testing

    Data<FunctionType> points(&coords);
    geometry.setData(&points);


    // Now we create the segmentation. One way of doing this is to store a set of sets
    // style representation. Here "segment" 0 will correspond to all point indices part
    // of feature/node 0 etc.
    vector<vector<GlobalIndexType> > segments(num_nodes);

    // In this case let's assume only the leafs contain data (like in a clustering hierarchy)

    // Leafs are same as the geometry points added
    for (int i=0;i<num_points;i++)
      segments[i].push_back(i);


    // And just like usual we now add this data to a segmentation handle
    SegmentationHandle segmentation;
    segmentation.encoding(true); // Make things ascii for testing
    segmentation.domainType(POINT_SET);

    char domain_description[100];
    sprintf(domain_description,"3 %f %f %f %f %f %f", 0.0,9.0,0.0,9.0,0.0,0.0);
    segmentation.domainDescription(domain_description);


    segmentation.setSegmentation(&segments);

    //Now we add the geometry to the segmentation
    segmentation.add(geometry);


    // Just for demonstration purposes we re-open the earlier file
    clan.attach(filename);

    // We happen to know that this is the first family in the file and we "append" our information
    // Note that this call will automatically write this information to file and update the meta data
    clan.family(0).append(segmentation);


    // We are done ... have a look at the bottom of the file to better understand what is happening





    return 1;
}
