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
package graph.basic;

import java.io.Serializable;

/**
 * This class represents graph edges.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DMGraphEdge implements Serializable {

    public double weight = 0;
    public int first;
    public int second;
    // Edges are maintained as linked lists.
    public DMGraphEdge next = null;
    public DMGraphEdge previous = null;

    /**
     * Empty constructor.
     */
    public DMGraphEdge() {
    }

    /**
     * Initialization.
     *
     * @param first Integer that is the index of the first vertex.
     * @param second Integer that is the index of the second vertex.
     */
    public DMGraphEdge(int first, int second) {
        this.first = first;
        this.second = second;
    }

    /**
     * Initialization.
     *
     * @param first Integer that is the index of the first vertex.
     * @param second Integer that is the index of the second vertex.
     * @param weight Double that is the edge weight.
     */
    public DMGraphEdge(int first, int second, double weight) {
        this.first = first;
        this.second = second;
        this.weight = weight;
    }

    /**
     * Creates the copy of the current edge.
     *
     * @return DMGraphEdge that is the copy of the current edge.
     */
    public DMGraphEdge copy() {
        DMGraphEdge copy = new DMGraphEdge();
        copy.weight = weight;
        copy.first = first;
        copy.second = second;
        copy.next = next;
        copy.previous = previous;
        return copy;
    }
}