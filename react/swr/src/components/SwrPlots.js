import React, { Component } from 'react';
import SwrPlotSpend from './SwrPlotSpend';
import SwrPlotPortval from './SwrPlotPortval';
import SwrPlotProfile from './SwrPlotProfile';
import SwrPlotSurvival from './SwrPlotSurvival';

class SwrPlots extends Component {

  constructor(props) {
    super(props);
    this.state = {
    };
  }

  render() {
    return (
      <div className="container">
        <div className="row ">
          <div className="swrplots-custom">
              <h5 className="plot-title">Spending Path for Retirement Cohorts 1928-1991</h5>
              <SwrPlotSpend mainObj={this.props.mainObj} />
          </div>
          <div className="swrplots-custom">
              <h5 className="plot-title">Portfolio Path for Retirement Cohorts 1928-1991</h5>
              <SwrPlotPortval mainObj={this.props.mainObj} />
          </div>
          <div className="swrplots-custom">
              <h5 className="plot-title">Spending v. Portfolio Value</h5>
              <SwrPlotProfile mainObj={this.props.mainObj} />
          </div>
          <div className="swrplots-custom">
              <h5 className="plot-title">Exhaustion Frequency</h5>
              <SwrPlotSurvival mainObj={this.props.mainObj} />
          </div>
        </div>
      </div>
    );
  }
}

export default SwrPlots;