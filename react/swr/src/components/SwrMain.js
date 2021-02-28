import React, { Component } from 'react';
import SwrHeader from './SwrHeader';
import SwrForm from './SwrForm';
import SwrDescription from './SwrDescription';
import SwrPlots from './SwrPlots';
import SwrFooter from './SwrFooter';
import { SPEND_DF } from '../js/spend_df.js';
import { PORTVAL_DF } from '../js/portval_df.js';
import { PROFILE_DF } from '../js/profile_df.js';
import { SURVIVAL_DF } from '../js/survival_df.js';
    
export class SwrMain extends Component {
    constructor(props) {
        super(props);
        this.state = {
            stock_alloc_pct: 50.0,
            bond_alloc_pct: 50.0,
            withdrawal_fixed_pct: 2.0,
            withdrawal_variable_pct: 2.5,
            withdrawal_floor_pct: 3.5,
            spend_df: SPEND_DF,
            portval_df: PORTVAL_DF,
            profile_df: PROFILE_DF,
            survival_df: SURVIVAL_DF,
            plot4x: [1, 2, 3],
            plot4y: [2, 1, 3]      
        };

        this.handleStockChange = this.handleStockChange.bind(this);
        this.handleBondChange = this.handleBondChange.bind(this);
        this.handleFixedChange = this.handleFixedChange.bind(this);
        this.handleVariableChange = this.handleVariableChange.bind(this);
        this.handleFloorChange = this.handleFloorChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
    }

    handleSubmit(event) {
        // console.log('Current State is: ' + JSON.stringify(this.state));
        event.preventDefault();
    }

    handleStockChange(event) {
        this.setState({stock_alloc_pct:  event.target.value});
    }

    handleBondChange(event) {
        this.setState({bond_alloc_pct:  event.target.value});
    }

    handleFixedChange(event) {
        this.setState({withdrawal_fixed_pct:  event.target.value});
    }

    handleVariableChange(event) {
        this.setState({withdrawal_variable_pct:  event.target.value});
    }

    handleFloorChange(event) {
        this.setState({withdrawal_floor_pct:  event.target.value});
    }

    render() {
        return (
            <div>
                <SwrHeader />
                <SwrForm mainObj={this} />
                <SwrDescription {...this.state} />
                <SwrPlots mainObj={this} />
                <SwrFooter />
            </div>
        );
    }
}

export default SwrMain;
