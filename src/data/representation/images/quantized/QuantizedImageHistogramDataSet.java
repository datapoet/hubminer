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
import data.representation.util.DataMineConstants;
import java.util.ArrayList;

/**
 * Data holder class for a list of quantized SIFT histograms in an image
 * dataset.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QuantizedImageHistogramDataSet extends DataSet {

    /**
     * @param codebookSize Integer that is the size of the codebook used for
     * quantization and hence the length of the individual histograms.
     */
    public QuantizedImageHistogramDataSet(int codebookSize) {
        if (codebookSize < 0 || !DataMineConstants.isAcceptableInt(
                codebookSize)) {
            iAttrNames = null;
        }
        iAttrNames = new String[codebookSize];
        for (int i = 0; i < codebookSize; i++) {
            iAttrNames[i] = "codebook:" + i;
        }
    }

    /**
     * @param dset DataSet object to initialize the data definitions.
     */
    public QuantizedImageHistogramDataSet(DataSet dset) {
        iAttrNames = dset.iAttrNames;
        fAttrNames = dset.fAttrNames;
        sAttrNames = dset.sAttrNames;
        data = new ArrayList<>(dset.size());
        for (int i = 0; i < dset.size(); i++) {
            data.add(dset.getInstance(i));
        }
    }
}
