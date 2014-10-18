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
package graph.drawing;

/**
 * This interface defines the interface for finding the proper coordinates for
 * the nodes in the graph, prior to display.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public interface CoordinateFinderInterface extends Runnable {

    /**
     * Whether to automatically updated the associated JGraph object.
     *
     * @param autoUpdateFlag Boolean flag determining the automatic JGraph
     * update.
     */
    public void setAutoJGUpdate(boolean autoUpdateFlag);

    /**
     * Finds the coordinates for the vertices in the graph.
     *
     * @throws Exception
     */
    public void findCoordinates() throws Exception;

    /**
     * Stops the process of finding the coordinates.
     */
    public void stop();

    /**
     * Gets the current progress of the calculations.
     *
     * @return Double that is the percentage of work done on calculating the
     * coordinates.
     */
    public double getProgress();
}
