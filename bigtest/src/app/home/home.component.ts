import { Component, OnInit } from '@angular/core';
import { Employee } from './employee';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Router } from '@angular/router';
import { ActivatedRoute } from '@angular/router';
import { EmployeeService } from '../employee.service';
@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {
  selectedManagerOption: number = 0;
  data = new Employee();
  currentDate: Date = new Date();
  dataFinal = new Employee();
  constructor(private employeeService:EmployeeService, private router : Router,private route : ActivatedRoute) { }

  ngOnInit(): void {

  }
  private padNumber(num: number): string {
    return num < 10 ? `0${num}` : `${num}`;
  }

  insertData(){

    const day = this.padNumber(this.currentDate.getDate());
    const month = this.padNumber(this.currentDate.getMonth() + 1);
    const year = this.currentDate.getFullYear();
    const hours = this.padNumber(this.currentDate.getHours());
    const minutes = this.padNumber(this.currentDate.getMinutes());
    const seconds = this.padNumber(this.currentDate.getSeconds());
    this.data.onboard_date= `${day}/${month}/${year} ${hours}:${minutes}:${seconds}`


    this.data.account_manager= this.selectedManagerOption

    this.employeeService.insert(this.data).subscribe(res=>{
      console.log(res)
     })
  }
}
