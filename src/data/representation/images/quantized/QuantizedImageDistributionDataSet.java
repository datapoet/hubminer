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
package data.representation.images.quantized;

import data.representation.DataSet;

/**
 * Data holder class for a list of quantized SIFT distributions in an image
 * dataset.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QuantizedImageDistributionDataSet extends DataSet {

    /**
     * @param codebookSize Integer that is the size of the codebook used for
     * quantization and hence the length of the individual histograms.
     */
    public QuantizedImageDistributionDataSet(int codebookSize) {
        fAttrNames = new String[codebookSize];
        for (int i = 0; i < codebookSize; i++) {
            fAttrNames[i] = "codebook:" + i;
        }
    }

    /**
     * @param dset DataInstance object to initialize the distribution with.
     */
    public QuantizedImageDistributionDataSet(DataSet dset) {
        fAttrNames = dset.fAttrNames;
        data = dset.data;
    }
}
