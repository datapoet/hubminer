/**
 * Hub Miner: a hubness-aware machine learning experimentation library.
 * Copyright (C) 2014 Nenad Tomasev. Email: nenad.tomasev at gmail.com
 * 
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */
package algref;

/**
 * Objects of this class hold the published info.
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class Publisher {
    
    private String name;
    private Address location;
    
    public static final Publisher SPRINGER = new Publisher("Springer-Verlag",
            new Address("Berlin", "Germany"));
    public static final Publisher IEEE = new Publisher("IEEE Computer Society",
            new Address("Washington", "USA"));
    public static final Publisher ACM = new Publisher("ACM",
            new Address("New York", "NY, USA"));
    public static final Publisher ELSEVIER = new Publisher(
            "Elsevier North-Holland, Inc.",
            new Address("Amsterdam", "The Netherlands"));
    public static final Publisher SIAM = new Publisher("SIAM/Omnipress",
            new Address("Philadelphia", "PA, USA"));
    public static final Publisher AAAI = new Publisher("AAAI Press",
            new Address("Menlo Park", "CA, USA"));
    public static final Publisher AAAS = new Publisher(
            "American Association for the Advancement of Science",
            new Address("New York", "NY, USA"));
    public static final Publisher ACADEMIC_PRESS = new Publisher(
            "Academic Press", new Address("Waltham", "MA, USA"));
    public static final Publisher MORGAN_KAUFMANN = new Publisher(
            "Morgan Kaufmann Publishers",
            new Address("San Francisco", "CA, USA"));
    public static final Publisher PERGAMON = new Publisher(
            "Pergamon Press, Inc.", new Address("Tarrytown", "NY, USA"));
    public static final Publisher KLUWER = new Publisher(
            "Kluwer Academic Publishers", new Address("Norwell", "MA, USA"));
    public static final Publisher INFORMATICA_SLOVENIA = new Publisher(
            "Slovenian Society Informatika",
            new Address("Ljubljana", "Slovenia"));
    
    /**
     * Default constructor.
     */
    public Publisher() {
    }
    
    /**
     * Initialization.
     * 
     * @param name String that is the publisher's name.
     * @param location String that is the publisher's location.
     */
    public Publisher(String name, Address location) {
        this.name = name;
        this.location = location;
    }

    /**
     * @return String that is the publisher's name.
     */
    public String getName() {
        return name;
    }

    /**
     * @param name String that is the publisher's name.
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * @return Address that is the publisher's location.
     */
    public Address getLocation() {
        return location;
    }

    /**
     * @param location Address that is the publisher's location.
     */
    public void setLocation(Address location) {
        this.location = location;
    }
}
