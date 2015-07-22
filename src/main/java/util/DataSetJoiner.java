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
package util;

import data.representation.DataSet;
import data.representation.images.sift.LFeatRepresentation;
import java.util.ArrayList;

/**
 * A utility class that joins several DataSet instances into one.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DataSetJoiner {

    /**
     * Joins an array of SIFTRepresentation instances into one.
     *
     * @param dsets An array of SIFTRepresentation objects.
     * @return SIFTRepresentation containing all objects from all individual
     * SIFTRepresentation-s in the array.
     */
    public static LFeatRepresentation joinSIFTCollections(
            LFeatRepresentation[] dsets) {
        if (dsets == null || dsets.length == 0) {
            return null;
        } else {
            LFeatRepresentation joinedDSet = new LFeatRepresentation(100, 10);
            if (dsets[0].identifiers != null) {
                DataSet[] identifiersArr = new DataSet[dsets.length];
                for (int i = 0; i < dsets.length; i++) {
                    if (dsets[i] != null) {
                        identifiersArr[i] = dsets[i].getIdentifiers();
                    }
                }
                joinedDSet.setIdentifiers(DataSetJoiner.join(identifiersArr));
            }
            int numAllInstances = 0;
            for (int i = 0; i < dsets.length; i++) {
                if ((dsets[i] != null) && !dsets[i].isEmpty()) {
                    numAllInstances += dsets[i].data.size();
                }
            }
            joinedDSet.data = new ArrayList<>(numAllInstances);
            for (int i = 0; i < dsets.length; i++) {
                if ((dsets[i] != null) && !dsets[i].isEmpty()) {
                    for (int j = 0; j < dsets[i].data.size(); j++) {
                        joinedDSet.addDataInstance(dsets[i].data.get(j));
                    }
                }
            }
            return joinedDSet;
        }
    }

    /**
     * Joins an array of DataSet objects into one.
     *
     * @param dsets Array of DataSet objects.
     * @return DataSet object that is joined from the passed array.
     */
    public static DataSet join(DataSet[] dsets) {
        if (dsets == null || dsets.length == 0) {
            return null;
        } else {
            DataSet joineDSet = new DataSet();
            joineDSet.fAttrNames = dsets[0].fAttrNames;
            joineDSet.sAttrNames = dsets[0].sAttrNames;
            joineDSet.iAttrNames = dsets[0].iAttrNames;
            if (dsets[0].identifiers != null) {
                DataSet[] identifiersArr = new DataSet[dsets.length];
                for (int i = 0; i < dsets.length; i++) {
                    if (dsets[i] != null) {
                        identifiersArr[i] = dsets[i].getIdentifiers();
                    }
                }
                joineDSet.setIdentifiers(DataSetJoiner.join(identifiersArr));
            }
            int numAllInstances = 0;
            for (int i = 0; i < dsets.length; i++) {
                if ((dsets[i] != null) && !dsets[i].isEmpty()) {
                    numAllInstances += dsets[i].data.size();
                }
            }
            joineDSet.data = new ArrayList<>(numAllInstances);
            for (int i = 0; i < dsets.length; i++) {
                if ((dsets[i] != null) && !dsets[i].isEmpty()) {
                    for (int j = 0; j < dsets[i].data.size(); j++) {
                        joineDSet.addDataInstance(dsets[i].data.get(j));
                    }
                }
            }
            return joineDSet;
        }
    }
}
