### Angular 2 
<hr />

<br />

#### Modules 
<hr />
An Angular module, whether a root or feature, is a class with an `@NgModule` decorator.

```typescript
// app/app.module.ts
import { NgModule }      from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
@NgModule({
  imports:      [ BrowserModule ],
  providers:    [ Logger ],
  declarations: [ AppComponent ],  
  exports:      [ AppComponent ],  // AppComponent is just for show, no modules will import from root
  bootstrap:    [ AppComponent ]   // They will import from AppComponent instead
})
export class AppModule { }
```

`NgModule` is a decorator function that takes a single metadata object whose properties describe the module. The most important are:

* **declarations** - the view classes that belong to this module. Angular has three kinds of view classes: *components*, *directives* and *pipes*.

* **exports** - subset of declarations that should be visible and usable in the component templates of other modules.

* **imports** - other modules whose exported classes are needed by component templates declared in this module.

* **providers** creators of services that this module contributes to the global collection of services; they become accessible in all parts of the app.

* **bootstrap** - identifies the main application view, called the root component, that hosts all other app views. Only the root module should set this bootstrap property.

<br />

#### Components 
<hr />
A **component** controls a patch of screen real estate that we could call a view.
 
``` typescript
 // app/hero-list.component.ts (class)
 export class HeroListComponent implements OnInit {
  heroes: Hero[];
  selectedHero: Hero;

  constructor(private service: HeroService) { }

  ngOnInit() {
    this.heroes = this.service.getHeroes();
  }

  selectHero(hero: Hero) { this.selectedHero = hero; }
}
```
<br />

<a name="Templates"></a>
#### Templates
<hr />
A **template** is a form of HTML that tells Angular how to render the component.

``` html
<!-- app/hero-list.component.html -->
<h2>Hero List</h2>
<p><i>Pick a hero from the list</i></p>
<ul>
  <li *ngFor="let hero of heroes" (click)="selectHero(hero)">
    {{hero.name}}
  </li>
</ul>
<hero-detail *ngIf="selectedHero" [hero]="selectedHero"></hero-detail>
```

<br />

#### Metadata
<hr />
**Metadata** tells Angular how to process a class.

For below example, in fact, it really is just a class. It's not a component until we tell Angular about it.

```typescript
@Component({
  selector:    'hero-list',
  templateUrl: 'app/hero-list.component.html',
  providers:   [ HeroService ]
})
export class HeroListComponent implements OnInit {
/* . . . */
}
```

Here are a few of the possible `@Component` configuration options:

* **selector**: CSS selector that tells Angular to create and insert an instance of this component where it finds a `<hero-list>` tag in parent HTML. For example, if an app's HTML contains `<hero-list></hero-list>`, then Angular inserts an instance of the HeroListComponent view between those tags.

* **templateUrl**: address of this component's template, which we showed [above](#Templates).

* **directives**: array of the components or directives that this template requires. We saw in the last line of our template that we expect Angular to insert a HeroDetailComponent in the space indicated by `<hero-detail>` **tags. Angular will do so only if we mention the HeroDetailComponent in this directives array.

* **providers**: array of *dependency injection* providers for services that the component requires. This is one way to tell Angular that our component's constructor requires a HeroService so it can get the list of heroes to display. We'll get to dependency injection later.

<br />

#### Data binding
<hr />
![data binding](databinding.png)


<br />