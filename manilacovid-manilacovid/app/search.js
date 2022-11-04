let search_el = document.getElementById("search-form");
let search_input = document.getElementById("searchInput");

let createCard = (content) => { return `<div class=" w-1/2 mx-auto lg:flex pb-5 pt-5">
<div class="h-48 lg:h-auto lg:w-48 flex-none bg-cover rounded-t lg:rounded-t-none lg:rounded-l text-center overflow-hidden" style="background-image: url(https://www.who.int/images/default-source/health-topics/coronavirus/corona-virus-getty.tmb-1024v.jpg?Culture=en&sfvrsn=217a6a68_6);" title="Woman holding a mug">
</div>
<div class="border-r border-b border-l border-grey-light lg:border-l-0 lg:border-t lg:border-grey-light bg-white rounded-b lg:rounded-b-none lg:rounded-r p-4 flex flex-col justify-between leading-normal">
  <div class="mb-8">    
    <p class="text-grey-darker text-base">${content}</p>
  </div>
</div>
</div>`}

search_el.addEventListener("submit", async function (evt) {
    evt.preventDefault();
    const api_url = "/api/chat";
    //const api_url = "https://manilacovid.info:8000/chat";
    const headers = {
        "Content-Type": "application/json"
    }
    try {
        const data = await postData(api_url, { "question": `What are the effects of covid19 on ${search_input.value}` });
        let recipe_el = document.getElementById("recipe_el");
        recipe_el.innerHTML = ""; 
        data.abstracts.forEach(element => {
            if(element[1].length > 100)
                recipe_el.innerHTML = recipe_el.innerHTML + createCard(element[1] )
        });
        
        console.log(JSON.stringify(data.abstracts)); // JSON-string from `response.json()` call
    } catch (error) {
        console.error(error);
    }
    async function postData(url = '', data = {}) {
        // Default options are marked with *
        const response = await fetch(url, {
            method: 'POST', // *GET, POST, PUT, DELETE, etc.
            mode: 'cors', // no-cors, *cors, same-origin
            cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
            credentials: 'same-origin', // include, *same-origin, omit
            headers: {
                'Content-Type': 'application/json'
                // 'Content-Type': 'application/x-www-form-urlencoded',
            },
            redirect: 'follow', // manual, *follow, error
            referrer: 'no-referrer', // no-referrer, *client
            body: JSON.stringify(data) // body data type must match "Content-Type" header
        });
        return await response.json(); // parses JSON response into native JavaScript objects
    }
});