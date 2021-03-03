import React, { Component } from 'react';
import SwrNavBar from './SwrNavBar';

class SwrHeader extends Component {
    render() {
      return(
      <React.Fragment>
        <SwrNavBar />
            <div className="container">
                 <div className="row row-header">
                     <div className="col-12 text-left">
                        <h1>Safe Withdrawal for Retirement Calculator</h1>
                        <p>Show all historical paths for 30-year retirements 1928-1991 based on specified asset allocation and withdrawal parameters.</p>
                        <p>Drag the sliders to adjust parameters:</p>
                        <ul>
                        <li>Asset allocation: Fixed stock/bond percentage throughout retirement.</li>
                        <li>Withdrawals:
                            <ul>
                                <li>Each year, withdraw a fixed percentage (Fixed %) of the starting portfolio <span class="font-italic">plus</span> a percentage of the current portfolio (Variable %), or</li>
                                <li>A floor percentage (Floor %) of the starting portfolio, whichever is greater</li>
                                <li>The spending profile chart illustrates spending for a given current portfolio.</li>
                            </ul>
                            </li>
                        </ul>
                        <p>See this <a href="https://druce.ai/2021/02/optimal-safe-withdrawal-for-retirement-using-certainty-equivalent-spending-revisited">blog post</a> for detailed discussion.</p>
	                    <p class="font-italic">Disclaimer: This is not investment advice! For educational purposes only. Past performance may not be representative of future results. Consult a professional for investment and retirement advice.</p>
                     </div>
                 </div>
             </div>
      </React.Fragment>
      );
    }
  }
  
  export default SwrHeader;