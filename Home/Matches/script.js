const rect = document.querySelector(".rect");
let today = new Date();

const options = {
  method: "GET",
  headers: {
    "X-RapidAPI-Key": "fa75f7cffbmsh2f16132e68bbfdap19ebe4jsnb8ea9113474e",
    "X-RapidAPI-Host": "footapi7.p.rapidapi.com",
  },
};
fetch(
  `https://footapi7.p.rapidapi.com/api/matches/top/${today.getDate()}/${
    today.getMonth() + 1
  }/${today.getFullYear()}`,
  options
)
  .then((response) => response.json())
  .then((response) => {
    for (let i = 0; i < 200; i++) {
      const htmlClubs = `<p class="clubs c${i}">${response.events[i].awayTeam.name} vs ${response.events[i].homeTeam.name}
      (${response.events[i].status.description})</p>`;
      rect.insertAdjacentHTML("beforeend", htmlClubs);
      const htmlScore = `<p class= "score">${
        response.events[i].homeScore.display
          ? response.events[i].homeScore.display
          : 0
      } - ${
        response.events[i].homeScore.display
          ? response.events[i].homeScore.display
          : 0
      }</p>`;
      document
        .querySelector(`.c${i}`)
        .insertAdjacentHTML("beforeend", htmlScore);
    }
  })
  .catch((err) => console.error(err));
// fetch("https://footapi7.p.rapidapi.com/api/team/2672/image", options)
//   .then((response) => response.json())
//   .then((response) => console.log(response))
//   .catch((err) => console.error(err));
