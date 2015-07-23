/**
* Hub Miner: a hubness-aware machine learning experimentation library.
* Copyright (C) 2014  Nenad Tomasev. Email: nenad.tomasev at gmail.com
* 
* This program is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the Free Software
* Foundation, either version 3 of the License, or (at your option) any later
* version.
* 
* This program is distributed in the hope that it will be useful, but WITHOUT
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with
* this program. If not, see <http://www.gnu.org/licenses/>.
*/
package data.representation.images.sift.util;

import data.representation.DataInstance;
import data.representation.images.sift.LFeatRepresentation;
import static data.representation.images.sift.LFeatRepresentation.
        DEFAULT_DESCRIPTOR_LENGTH;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusterConfigurationCleaner;

/**
 * A utility representation class similar to SIFTRepresentation, extended by an
 * integer value indicating the cluster affiliation of individual vectors.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ClusteredSIFTRepresentation extends LFeatRepresentation {

    /**
     *
     */
    public ClusteredSIFTRepresentation() {
        this(DEFAULT_DESCRIPTOR_LENGTH);
    }

    /**
     * @param descLength Descriptor length.
     */
    public ClusteredSIFTRepresentation(int descLength) {
        fAttrNames = new String[descLength + 4];
        fAttrNames[0] = "Y";
        fAttrNames[1] = "X";
        fAttrNames[2] = "Scale";
        fAttrNames[3] = "Angle";
        for (int i = 0; i < descLength; i++) {
            fAttrNames[i + 4] = "desc" + i;
        }
        iAttrNames = new String[1];
        iAttrNames[0] = "Cluster";
    }

    /**
     * @param sr SIFTRepresentation to base this clustered representation on.
     */
    public ClusteredSIFTRepresentation(LFeatRepresentation sr) {
        super();
        fAttrNames = sr.fAttrNames;
        iAttrNames = sr.iAttrNames;
        data = sr.data;
        identifiers = sr.identifiers;
        sAttrNames = sr.sAttrNames;
    }

    /**
     * Turn the ClusteredSIFTRepresentation object into an array of clusters.
     *
     * @return Cluster[] array, clusters defined by this clustered
     * representation.
     */
    public Cluster[] representAsClusters() {
        int max = -1;
        for (int i = 0; i < data.size(); i++) {
            int tmp = ((DataInstance) (data.get(i))).iAttr[0];
            if (tmp > max) {
                max = tmp;
            }
        }
        Cluster[] clusters = new Cluster[max + 1];
        for (int i = 0; i < clusters.length; i++) {
            clusters[i] = new Cluster(this, 150);
        }
        for (int i = 0; i < data.size(); i++) {
            clusters[(data.get(i)).iAttr[0]].addInstance(i);
        }
        clusters = ClusterConfigurationCleaner.removeEmptyClusters(clusters);
        for (int i = 0; i < clusters.length; i++) {
            for (int j = 0; j < clusters[i].size(); j++) {
                (clusters[i].getInstance(j)).iAttr[0] = i;
            }
        }
        return clusters;
    }

    /**
     * Get the ClusteredSIFTRepresentation based on the provided clusters.
     *
     * @param clusters Cluster[] of SIFTVector objects.
     * @return
     */
    public static ClusteredSIFTRepresentation getFromClusters(
            Cluster[] clusters) {
        if (clusters == null || clusters.length == 0) {
            return null;
        }
        ClusteredSIFTRepresentation rep = new ClusteredSIFTRepresentation(
                DEFAULT_DESCRIPTOR_LENGTH);
        Cluster[] nonEmptyClusters = ClusterConfigurationCleaner.
                removeEmptyClusters(clusters);
        DataInstance instance;
        for (int i = 0; i < nonEmptyClusters.length; i++) {
            for (int j = 0; j < nonEmptyClusters[i].size(); j++) {
                instance = nonEmptyClusters[i].getInstance(j);
                instance.embedInDataset(rep);
                instance.iAttr = new int[1];
                instance.iAttr[0] = i;
                rep.addDataInstance(instance);
            }
        }
        return rep;
    }
}