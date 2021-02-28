import React, { Component } from 'react';
import { Jumbotron } from 'reactstrap';
import SwrNavBar from './SwrNavBar';

class SwrHeader extends Component {
    render() {
      return(
      <React.Fragment>
        <SwrNavBar />
        <Jumbotron>
            <div className="container">
                 <div className="row row-header">
                     <div className="col-12">
                        <h1>Safe Withdrawal for Retirement Calculator</h1>
                        <p>Show paths for 30-year retirement based on parameters. See blog post.</p>
	                    <p>Disclaimer: This is not investment advice! For educational purposes only. Past performance may not be representative of future results. Consult a professional for investment and retirement advice.</p>
                     </div>
                 </div>
             </div>
         </Jumbotron>
      </React.Fragment>
      );
    }
  }
  
  export default SwrHeader;