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

import java.util.ArrayList;

/**
 * Objects of this class represent journal publications.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class JournalPublication extends Publication {
    
    private String journalName;
    private int volume;
    private int issue;
    
    /**
     * Default constructor.
     */
    public JournalPublication() {
    }
    
    /**
     * Initialization.
     * 
     * @param title String that is the publication title.
     * @param authors ArrayList<Author> of publication authors.
     * @param publisher Publisher that published the publication.
     * @param doi String that is the DOI.
     * @param url String that is the URL.
     * @param startPage Integer that is the start page.
     * @param endPage Integer that is the end page.
     * @param year Integer that is the year of the publication.
     */
    public JournalPublication(String title, ArrayList<Author> authors, 
            Publisher publisher, String doi, String url, int startPage,
            int endPage, int year) {
        super(title, authors, publisher, doi, url, startPage, endPage, year);
    }
    
    /**
     * Initialization.
     * 
     * @param title String that is the publication title.
     * @param authors ArrayList<Author> of publication authors.
     * @param publisher Publisher that published the publication.
     * @param doi String that is the DOI.
     * @param url String that is the URL.
     * @param startPage Integer that is the start page.
     * @param endPage Integer that is the end page.
     * @param year Integer that is the year of the publication.
     * @param journalName String representing the journal name.
     * @param volume Integer representing the volume.
     * @param issue Integer representing the issue.
     */
    public JournalPublication(String title, ArrayList<Author> authors, 
            Publisher publisher, String doi, String url, int startPage,
            int endPage, int year, String journalName, int volume, int issue) {
        super(title, authors, publisher, doi, url, startPage, endPage, year);
        this.journalName = journalName;
        this.volume = volume;
        this.issue = issue;
    }
    
    /**
     * @return String representing the journal name.
     */
    public String getJournalName() {
        return journalName;
    }

    /**
     * @param journalName String representing the journal name.
     */
    public void setJournalName(String journalName) {
        this.journalName = journalName;
    }

    /**
     * @return Integer representing the journal volume.
     */
    public int getVolume() {
        return volume;
    }

    /**
     * @param volume Integer representing the journal volume.
     */
    public void setVolume(int volume) {
        this.volume = volume;
    }

    /**
     * @return Integer representing the journal issue.
     */
    public int getIssue() {
        return issue;
    }

    /**
     * @param issue Integer representing the journal issue.
     */
    public void setIssue(int issue) {
        this.issue = issue;
    }
    
}
