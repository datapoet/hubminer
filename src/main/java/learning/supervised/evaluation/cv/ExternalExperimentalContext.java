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
package learning.supervised.evaluation.cv;

import data.neighbors.NeighborSetFinder;

/**
 * Objects of this class may represent some externally-calculated useful shared
 * objects, like the distance matrix and the primary k-nearest neighbor sets.
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ExternalExperimentalContext {
    
    // The distance matrix on the entire data.
    private float[][] primaryDMat;
    // kNN sets in the primary metric, on the entire data.
    private NeighborSetFinder primaryNSF;
    
    /**
     * Initialization.
     */
    public ExternalExperimentalContext() {
    }
    
    /**
     * Initialization.
     * 
     * @param primaryDMat float[][] representing the distance matrix on the
     * entire data.
     * @param primaryNSF NeighborSetFinder object holding the kNN sets
     * calculated in the primary metric, on the entire data, for the largest
     * k-value that is needed.
     */
    public ExternalExperimentalContext(float[][] primaryDMat,
            NeighborSetFinder primaryNSF) {
        this.primaryDMat = primaryDMat;
        this.primaryNSF = primaryNSF;
    }
    
    /**
     * @return float[][] representing the distance matrix on the entire data.
     */
    public float[][] getDistances() {
        return primaryDMat;
    }
    
    /**
     * @param primaryDMat float[][] representing the distance matrix on the
     * entire data.
     */
    public void setDistances(float[][] primaryDMat) {
        this.primaryDMat = primaryDMat;
    }
    
    /**
     * @return NeighborSetFinder object holding the kNN sets calculated in the
     * primary metric, on the entire data, for the largest k-value that is
     * needed. 
     */
    public NeighborSetFinder getNeighborSets() {
        return primaryNSF;
    }
    
    /**
     * @param primaryNSF NeighborSetFinder object holding the kNN sets
     * calculated in the primary metric, on the entire data, for the largest
     * k-value that is needed. 
     */
    public void setNeighborSets(NeighborSetFinder primaryNSF) {
        this.primaryNSF = primaryNSF;
    }
    
}
