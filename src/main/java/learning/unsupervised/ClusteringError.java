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
package learning.unsupervised;

/**
 * Error messages pertaining to the clustering process.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ClusteringError extends Exception {

    public static final int UNKNOWN_PROBLEM = 0;
    public static final int EMPTY_CLUSTER = 1;
    // The algorithms might throw this if they cannot produce a configuration
    // after many attempts and restarts.
    public static final int UNABLE_TO_FINISH = 2;
    private int cause = UNKNOWN_PROBLEM;

    /**
     * @param cause An integer indicating what was the problem.
     */
    public ClusteringError(int cause) {
        this.cause = cause;
    }

    /**
     * @return Cause of the exception.
     */
    public int getErrorCause() {
        return cause;
    }
}