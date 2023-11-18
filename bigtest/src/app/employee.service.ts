import { Injectable } from '@angular/core';
import {HttpClient} from '@angular/common/http';
import { Router } from '@angular/router';
@Injectable({
  providedIn: 'root'
})
export class EmployeeService {

  constructor(private httpClient: HttpClient,private router: Router) { }
  insert(data : any ){

    return this.httpClient.post('http://localhost:5000/data',data);

  }
}
