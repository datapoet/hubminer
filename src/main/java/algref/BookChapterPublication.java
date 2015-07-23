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
 * Objects of this class represent book chapter publications. The title of the 
 * chapter is represented by the title field in the parent class and this class 
 * adds the book name as additional information.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BookChapterPublication extends Publication {
    
    private String bookName;
    
    /**
     * Default constructor.
     */
    public BookChapterPublication() {
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
    public BookChapterPublication(String title, ArrayList<Author> authors, 
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
     * @param bookeName String representing the book name.
     */
    public BookChapterPublication(String title, ArrayList<Author> authors, 
            Publisher publisher, String doi, String url, int startPage,
            int endPage, int year, String bookName) {
        super(title, authors, publisher, doi, url, startPage, endPage, year);
        this.bookName = bookName;
    }

    /**
     * @return String representing the book name.
     */
    public String getBookName() {
        return bookName;
    }

    /**
     * @param conferenceName String representing the book name.
     */
    public void setBookName(String bookName) {
        this.bookName = bookName;
    }
    
}
