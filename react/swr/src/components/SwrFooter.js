import React from 'react';
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faRssSquare } from "@fortawesome/free-solid-svg-icons";
import { faTwitterSquare, faGithubSquare, faLinkedin } from "@fortawesome/free-brands-svg-icons";
import '../css/SwrFooter.css';

function SwrFooter(props) {
    return(
    <div className="footer">
        <div className="container">
            <div className="row justify-content-center">             
                <div className="col-auto footer-custom">
                    &copy; Copyright 2021 <a className="footer-custom" href="https://druce.ai/contact.html">Druce Vertes</a>
                    <br />
                    <a href="https://druce.ai/feed.xml" style={{color: "#ee802f"}} target="_blank" rel="noreferrer">
                    <FontAwesomeIcon icon={faRssSquare} />
                    </a>
                    &nbsp;
                    <a href="https://twitter.com/streeteye/" style={{color: "#1DA1F2"}} target="_blank" rel="noreferrer">
                        <FontAwesomeIcon icon={faTwitterSquare} />
                    </a>
                    &nbsp;
                    <a href="https://github.com/druce" style={{color: "#6e5494"}} target="_blank" rel="noreferrer">
                    <FontAwesomeIcon icon={faGithubSquare} />
                    </a>
                    &nbsp;
                    <a href="https://www.linkedin.com/in/drucevertes" style={{color: "#0e76a8"}}target="_blank" rel="noreferrer">
                        <FontAwesomeIcon icon={faLinkedin} />
                    </a>                
                </div>
            </div>
        </div>
    </div>
    )
}

export default SwrFooter;