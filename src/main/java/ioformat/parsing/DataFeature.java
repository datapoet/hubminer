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
package ioformat.parsing;

/**
 * This class implements the auxiliary logic for attribute parsing and modeling.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DataFeature {

    private String featureName;
    private int featureType;
    private int featureIndex;

    /**
     * @return String that is the feature name.
     */
    public String getFeatureName() {
        return featureName;
    }

    /**
     * @param name String that is the feature name.
     */
    public void setFeatureName(String name) {
        this.featureName = name;
    }

    /**
     * @return Integer that is the feature index.
     */
    public int getFeatureIndex() {
        return featureIndex;
    }

    /**
     * @param index Integer that is the feature index.
     */
    public void setFeatureIndex(int index) {
        this.featureIndex = index;
    }

    /**
     * @return Integer that is the feature type, as in DataMineConstants.
     */
    public int getFeatureType() {
        return featureType;
    }

    /**
     * @param type Integer that is the feature type, as in DataMineConstants.
     */
    public void setFeatureType(int type) {
        this.featureType = type;
    }
}
