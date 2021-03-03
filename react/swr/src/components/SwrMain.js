import React, { Component } from 'react';
import SwrHeader from './SwrHeader';
import SwrSliders from './SwrSliders';
import SwrDescription from './SwrDescription';
import SwrPlots from './SwrPlots';
import SwrFooter from './SwrFooter';
import { REAL_RETURN_DF } from '../js/real_return_df.js';
import { SPEND_DF } from '../js/spend_df.js';
import { PORTVAL_DF } from '../js/portval_df.js';
import { PROFILE_DF } from '../js/profile_df.js';
import { SURVIVAL_DF } from '../js/survival_df.js';
import SwrQuickBtns from './SwrQuickBtns';

// sort arr1 per order in arr2
// const dsu = (arr1, arr2) => {
    // return arr1
    //   .map((item, index) => [arr2[index], item])
    //   .sort(([arg1], [arg2]) => arg2 - arg1)
    //   .map(([, item]) => item); 
    // }
    
export class SwrMain extends Component {
    constructor(props) {
        super(props);
        this.state = {
            stock_alloc_pct: 50.0,
            bond_alloc_pct: 50.0,
            withdrawal_fixed_pct: 2.0,
            withdrawal_variable_pct: 2.5,
            withdrawal_floor_pct: 3.5,
            n_ret_years: 30,
            real_return_df: REAL_RETURN_DF,
            spend_df: SPEND_DF,
            portval_df: PORTVAL_DF,
            profile_df: PROFILE_DF,
            survival_df: SURVIVAL_DF,
        };

        this.handleStockChange = this.handleStockChange.bind(this);
        this.handleBondChange = this.handleBondChange.bind(this);
        this.handleFixedChange = this.handleFixedChange.bind(this);
        this.handleVariableChange = this.handleVariableChange.bind(this);
        this.handleFloorChange = this.handleFloorChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
        this.calc_profile_df = this.calc_profile_df.bind(this);
        this.recalc_cohort_data = this.recalc_cohort_data.bind(this);

        this.calc_profile_df();
        this.recalc_cohort_data();
    }

    calc_profile_df() {
        let portvals = [];
        let spendvals = []
        let withdrawal_variable = this.state.withdrawal_variable_pct / 100;
        for (var i=0; i<=2000; i++) {
            let portval = i/10;
            let desired_spend = this.state.withdrawal_fixed_pct + (portval * withdrawal_variable);
            let desired_spend2 = Math.max(desired_spend, this.state.withdrawal_floor_pct);
            let spendval = Math.min(portval, desired_spend2);
            portvals.push(portval);
            spendvals.push(spendval);
        }
        // copy profile from state, update and return it
        let new_df = Object.assign(this.state.profile_df);
        new_df.data = [portvals, spendvals];
        this.setState({profile_df: new_df});
     }

    recalc_cohort_data() {
        let stocks = this.state.real_return_df.data[0];
        let bonds = this.state.real_return_df.data[1];
        let stock_alloc = this.state.stock_alloc_pct / 100;
        let bond_alloc = this.state.bond_alloc_pct / 100;
        let withdrawal_variable = this.state.withdrawal_variable_pct / 100
        let n_ret_years = this.state.n_ret_years;
        let n_cohorts = stocks.length - n_ret_years + 1;
        let spendvaldata = []
        let portvaldata = []
        // should use n_cohorts to build correct number
        let new_survival_df = {
            "columns":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
            "index":["survival_count","survival_bins"],
            "data":[
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
            ]
        }

        for (let cohort_start=0; cohort_start<n_cohorts; cohort_start++) {
            let portval = 100.0;
            let portvals = [];
            let spendvals = [];
            portvals.push(portval);
            let exhausted = false;
            for (let cohort_year=0; cohort_year<n_ret_years; cohort_year++) {
                let current_year = cohort_start + cohort_year;
                let real_return = stock_alloc * stocks[current_year] + bond_alloc * bonds[current_year];
                portval *= ( 1 + real_return );
                let desired_spend = this.state.withdrawal_fixed_pct + portval * withdrawal_variable;
                desired_spend = Math.max(desired_spend, this.state.withdrawal_floor_pct);
                let spendval = Math.min(portval, desired_spend);
                spendvals.push(spendval);
                portval -= spendval;
                portvals.push(portval);
                if (portval === 0) {
                    if (!exhausted) {
                        exhausted = true;
                        new_survival_df.data[0][cohort_year]++;
                    }
                }
            }
            spendvaldata.push(spendvals);
            portvaldata.push(portvals);
            if (!exhausted) {
                new_survival_df.data[0][n_ret_years]++;
            }
        }
        // resort per ending portval
        // let final_portvals = [];
        // let final_portval_indexes = [];        
        // for (let i = 0; i < portvaldata.length; i++) {
        //     final_portval_indexes.push(i);
        //     final_portvals.push(portvaldata[i][portvaldata[i].length-1]);
        // }
        // console.log(final_portvals);
        // let new_order = dsu(final_portval_indexes, final_portvals);
        // console.log(new_order);

        // let new_spendvaldata = [];
        // let new_portvaldata = [];
        // for (let i = 0; i < new_order.length; i++) {
        //     new_spendvaldata.push(spendvaldata[new_order[i]]);
        //     new_portvaldata.push(portvaldata[new_order[i]]);
        // }

        let new_spend_df = Object.assign(this.state.spend_df);
        new_spend_df.data = spendvaldata;
        this.setState({spend_df: new_spend_df});

        let new_portval_df = Object.assign(this.state.portval_df);
        new_portval_df.data = portvaldata;
        this.setState({portval_df: new_portval_df});

        this.setState({survival_df: new_survival_df});
        
    }

    handleSubmit(event) {
        event.preventDefault();
        this.calc_profile_df();
        this.recalc_cohort_data();

        // console.log('Current State is: ' + JSON.stringify(this.state));
    }

    handleStockChange(event) {
        let targetval = parseFloat(event.target.value)
        targetval = isNaN(targetval) ? 0 : targetval;
        this.setState({stock_alloc_pct: targetval});
    }

    handleBondChange(event) {
        let targetval = parseFloat(event.target.value)
        targetval = isNaN(targetval) ? 0 : targetval;
        this.setState({bond_alloc_pct: targetval});
    }

    handleFixedChange(event) {
        let targetval = parseFloat(event.target.value)
        targetval = isNaN(targetval) ? 0 : targetval;
        this.setState({withdrawal_fixed_pct: targetval});
    }

    handleVariableChange(event) {
        let targetval = parseFloat(event.target.value)
        targetval = isNaN(targetval) ? 0 : targetval;
        this.setState({withdrawal_variable_pct: targetval});
    }

    handleFloorChange(event) {
        let targetval = parseFloat(event.target.value)
        targetval = isNaN(targetval) ? 0 : targetval;
        this.setState({withdrawal_floor_pct: targetval});
    }

    render() {
        return (
            <div>
                <SwrHeader />
                <SwrQuickBtns mainObj={this} />
                <SwrSliders mainObj={this} />
                <SwrDescription {...this.state} />
                <SwrPlots mainObj={this} />
                <SwrFooter />
            </div>
        );
    }
}

export default SwrMain;
