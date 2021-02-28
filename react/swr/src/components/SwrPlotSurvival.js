// import { data } from 'jquery';
import React, { Component } from 'react';
import Plot from 'react-plotly.js';

class SwrPlotSurvival extends Component {

  render() {

    let df = this.props.mainObj.state.survival_df;
    let plotly_data = [{
        x: df.data[1],
        y: df.data[0],
        type: 'bar',
      }];

    return (
      <Plot
        data={plotly_data}
        layout={{width: 480,
          height: 300, 
          showlegend: false,
          title: false,
          yaxis: {title: 'Count',
            linecolor: 'black',
            mirror: true,
            ticks: 'inside'},
          xaxis: {title: 'Years to Exhaustion',
            linecolor: 'black',
            mirror: true,
            ticks: 'inside'},
          margin: {
            t: 5,
            r: 10,
            l: 50,
            b: 40,
            pad: 0
          }}
        }
      />
    );
  }
}

export default SwrPlotSurvival;