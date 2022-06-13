#include "identitiestracker.h"


bool IdentitiesTracker::Exists(long id) const {
    return data_.end() != data_.find(id);
}

void IdentitiesTracker::Update(long id, IdentityStatus status){
    if (data_.find(id) == data_.end()) {
        Identity identity;
        identity.status = status;
        data_.insert({id, identity});
    }
    data_.at(id).status=status;
}

IdentityStatus  IdentitiesTracker::GetStatus(long id) const {
    return data_.at(id).status;
}
