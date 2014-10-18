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
package sampling;

import data.representation.DataSet;

/**
 * The basic sampling abstract class.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public abstract class Sampler {

    private boolean repetitions = false;

    /**
     * @param repetitions Boolean flag denoting whether to use repetitive
     * sampling or not.
     */
    public Sampler(boolean repetitions) {
        this.repetitions = repetitions;
    }

    /**
     * @param repetitions Boolean flag denoting whether to use repetitive
     * sampling or not.
     */
    public void setRepetitions(boolean repetitions) {
        this.repetitions = repetitions;
    }

    /**
     * @return Boolean value indicating whether the sampler uses repetitive
     * sampling or not.
     */
    public boolean getRepetitions() {
        return repetitions;
    }

    /**
     * Gets a uniform subsample.
     *
     * @param dset DataSet object to be sampled from.
     * @param sampleSize Size of the desired sample.
     * @return DataSet object containing the subsample of the original data.
     * @throws Exception
     */
    public abstract DataSet getSample(DataSet dset, int sampleSize)
            throws Exception;

    /**
     * Gets a uniform subsample.
     *
     * @param dsets DataSet object array to be sampled from.
     * @param sampleSize Size of the desired sample.
     * @return DataSet object containing the subsample of the original data.
     * @throws Exception
     */
    public abstract DataSet getSample(DataSet[] dsets, int sampleSize)
            throws Exception;
}
